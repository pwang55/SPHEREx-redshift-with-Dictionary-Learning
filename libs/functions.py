import numpy as np
from numba import jit, njit
from numba.typed import List
from math import erfc


c = 3e18
msqrt1_2 = 1/np.sqrt(2)


# function to convert f_nu to f_lambda 
def fnu2flambda(wl, flux):
    flux = flux * c/wl**2
    return flux

# function to convert f_lambda to f_nu
def flambda2fnu(wl, flux):
    flux = flux * wl**2/c
    return flux

# Multivariate gaussian integral
def multivariate_gaussian_int(cov):
    ndim = cov.shape[0]
    return (2*np.pi)**(ndim/2)*np.sqrt(np.linalg.det(cov))

# Function for linear interpolation with 2-d array
@jit(nopython=True, fastmath=True)
def multiInterp1(x, xp, fp):
    '''
    Interpolate 2-D matrix fp along axis=1 at point x
    Caution: a very long x array might make it slower than using np.interp with for loop

    Parameters
    ----------
    x : ndarray of shape n
        Data points to interpolate
    
    xp : ndarray of shape np
        X coordinate points for matrix fp in axis=1
    
    fp : ndarray of shape (nf, np)
        2-D matrix containing nf rows of Y values along axis=1
    
    Returns
    -------
    f : ndarray of shape (nf, n)
        Interpolated 2-D matrix
        
    '''
    ji = np.searchsorted(xp, x) - 1
    jf = ji+1
    xj = (x-xp[ji])/(xp[jf]-xp[ji])
    f = (1-xj) * fp[:,ji] + xj * fp[:,jf]
    return f


# OLS function with simple X, y input format
@jit(nopython=True, fastmath=True)
def ols(X, y):
    coef = np.linalg.inv(X.T @ X) @ X.T @ y
    # model = X @ params
    return coef


@jit(nopython=True, fastmath=True)
def larslasso(X, y, alpha=0.0, positive=False, center_Xy=False, unit_X=True, unit_y=True, path=False, best_estimator=None,
                max_iter=200, decimals=10, alpha_ns_scaling=False, max_feature=None, active_ols=False):
    '''
    Return LARS-lasso fitting coefficients. X is the atoms array, y is the target vector.

    Parameters
    ----------
    X : ndarray of shape (ns, nf)
        Atom array with ns samples and nf features

    y : ndarray of shape (ns, )
        Target vector with ns sample points to fit

    alpha : float or ndarray of shape (nf, ), default=0.0
        Regularization parameter for LARS-lasso algorithm. alpha>=0. Alpha=0 will yield OLS solution.
        The actual mathematical meaning of this parameter is the lowest required dot product value between atoms and target.
        If an ndarray is given, use each elemeht of alpha as minimum required correlation for each feature in X
    
    positive : bool, default=False
        If True, force all coefficients to be positive

    center_Xy : bool, default=False
        If True, center all the atoms in X and y

    unit_X : bool, default=True
        If False, do not normalize the input atoms
    
    unit_y : bool, default=True
        If False, do not normalize the input target vector

    path : bool, default=False
        If True, return coef, coef_path and alphas

    best_estimator : str, default=None
        Use best estimator fit as output, can be 'Cp' or 'BIC'

    best_bic : bool, default=False
        If True, output the best fit based on BIC before early stopping due to input alpha. Has priority over best_cp
        
    max_iter : int, default=200
        Max allowed iteration for the LARS-lasso algorithm

    decimals : int, default=10
        Number of decimals to round coefficients for stability

    alpha_ns_scaling : bool, default=False
        If True, alpha_i in each step will be alpha_i/ns 
        This is so that alpha is more consistent with lasso L1-regularization definition.

    max_feature : int, default=None
        Max allowed activated features before ending algorithm early
        If None, max_feature = nf

    active_ols : bool, default=False
        If True, only use alpha as active set selection and calculate ordinary least-square fit with active set when selction ends

    Returns
    -------
    coef : ndarray of shape (nf, )
        Fitted coefficients
    
    b : float
        Fitted incercept; the recovered model should be X @ coef + b
    
    coef_path : ndarray of shape (n_iters, nf)
        All coefficients along fitting path, will return np.array([[0.0]]) if path=False
    
    alpha_path : ndarray of shape (n_iters, )
        All alpha values along fitting path, will return np.array([0.0]) if path=False

    est_path : ndarray of shape (n_iters, )
        All estimator values along fitting path, will return np.array([0.0]) if path=False, and np.array([0, 0, ... 0]) with shape (n_iters) if best_estimator=None

    '''
    # X = np.ascontiguousarray(X)
    # y = np.ascontiguousarray(y)

    nf = X.shape[1]
    ns = X.shape[0]
    if max_feature is None:
        max_feature = nf

    if best_estimator is not None:
        path = True

    Xnorms = np.ones(nf)
    ynorm = 1.0
    Xmeans = np.zeros(nf)
    ymean = 0.0

    if center_Xy:
        Xmeans = np.sum(X, axis=0)/X.shape[0]    # numba doesn't support np.mean(X, axis=0)
        X = X - Xmeans
        ymean = np.mean(y)
        y = y - ymean
    if unit_X:
        Xnorms = np.sqrt(np.sum(X**2, axis=0))  # norm need to be re-calculated if centered
        # Xnorms[Xnorms==0.0] = 1.0
        X = X / Xnorms
    if unit_y:
        ynorm = np.linalg.norm(y)
        y = y / ynorm

    Cov = X.T @ y
    G = X.T @ X

    coef = np.zeros(nf)
    prev_coef = np.zeros(nf)
    alphas = np.array([0.0])
    prev_alphas = np.array([0.0])
    coef_path = coef.copy()[None,:]
    # alpha_path = alphas.copy()
    alpha_path = np.empty(shape=0)

    # check if input alpha is a single float number
    # set alpha_array for each feature, even if they are all the same
    # if best_cp:
        # alpha_array = np.zeros(nf)
    # else:
    alpha_array = alpha - np.zeros(nf)
    # alpha_array0 = alpha_array.copy()

    n_iter = 0
    nA = 0
    indices = np.arange(nf) # index number of atoms, always track the true index number with this
    idx_A_tf = np.full(nf, False)   # active atoms selection mask
    idx_rest_tf = np.full(nf, True) # non-active atoms mask
    sign_active = np.zeros(nf)      # active atoms will have 1 or -1

    drop = False
    Cov_copy = Cov.copy()
    G_copy = G.copy()

    while True:
        if Cov[idx_rest_tf].size:
            if positive:
                c_idx_arg = np.argmax(Cov[idx_rest_tf])              # index number for current Cov selection
            else:
                c_idx_arg = np.argmax(np.fabs(Cov[idx_rest_tf]))     # index number for current Cov selection
            c_idx = indices[idx_rest_tf][c_idx_arg]             # find the true index number
            c_ = np.max(np.fabs(Cov[idx_rest_tf]))
            c_ = Cov[c_idx]
            if positive:
                c = c_
            else:
                c = np.fabs(c_)
        else:
            c = 0.0

        if alpha_ns_scaling:
            alphas[0] = c/ns
        else:
            alphas[0] = c
        if path:
            alpha_path = np.hstack((alpha_path, alphas))

        # Early stopping criteria
        if (alphas < alpha_array[c_idx] and n_iter > 0) or (nA >= max_feature and n_iter > 0):
            if not active_ols:
                # if active_alpha_0:
                    # alpha_array = np.zeros_like(alpha_array)
                ss = (prev_alphas - alpha_array) / (prev_alphas - alphas)
                coef = prev_coef + ss * (coef - prev_coef)
                alphas[0] = alpha_array[c_idx]
            else:
                coef_A = ols(X[:, idx_A_tf], y)
                coef[idx_A_tf] = coef_A
            if path:
                # coef_path = np.vstack((coef_path, coef))
                # coef_path_copy = coef_path.copy()
                # coef_path = np.zeros((coef_path_copy.shape[0]+1, coef_path_copy.shape[1]))
                # coef_path[:-1] = coef_path_copy
                # coef_path[-1] = coef
                coef_path[-1] = coef
                # if active_alpha_0_fitting_penalty and active_alpha_0:
                    # alpha_path[-1] = alpha_array0[c_idx]
                # else:
                    # alpha_path[-1] = alpha_array[c_idx]
                alpha_path[-1] = alpha_array[c_idx]
                # coef_path = np.append(coef_path, coef).reshape((n_iter+2, nf))
            # else:
                # coef_path = coef_path[None,:]   # add an axis so that dimension and type is compatible with coef_path when path=True
            break

        if n_iter >= max_iter or nA >= nf:
            # print('Stopping criteria met')
            break

        if not drop:
            if positive:
                sign_active[c_idx] = np.ones_like(c_)
            else:
                sign_active[c_idx] = np.sign(c_)
            idx_A_tf[c_idx] = True
            idx_rest_tf[c_idx] = False
            nA += 1

        Cov_rest = Cov[idx_rest_tf]
        XA = X[:, idx_A_tf]
        Xr = X[:, idx_rest_tf]
        GA = XA.T @ XA
        one_A = sign_active[idx_A_tf]

        AA = 1/np.sqrt(one_A.T @ np.linalg.inv(GA) @ one_A)
        wA = AA * np.linalg.inv(GA) @ one_A
        uA = XA @ wA
        corr_eq_dir = Xr.T @ uA

        g1 = (c-Cov_rest)/(AA-corr_eq_dir)
        g1[g1<=0] = np.inf
        if g1.size == 0:
            g1 = np.array([np.inf])
        if positive:
            g = min(min(g1[g1>0]), c/AA)
        else:
            g2 = (c+Cov_rest)/(AA+corr_eq_dir)
            # print(g2)
            g2[g2<=0] = np.inf
            if g2.size == 0:
                g2 = np.array([np.inf])
            g = min(min(g1[g1>0]), min(g2[g2>0]), c/AA)

        drop = False
        z = -coef[idx_A_tf]/wA
        if z[z>0].size>0 and np.min(z[z>0]) < g:
            # print('Drop!')
            drop = True
            min_pos_z = np.min(z[z>0])
            idx_drop1 = np.where(z==min_pos_z)[0]
            idx_drop = indices[idx_A_tf][idx_drop1]
            g = min_pos_z

        n_iter += 1
        prev_coef = coef
        prev_alphas[0] = alphas[0]
        coef = np.zeros_like(coef)
        coef[idx_A_tf] = prev_coef[idx_A_tf] + g * wA
        Cov[idx_rest_tf] = Cov[idx_rest_tf] - g * corr_eq_dir

        if path:
            # coef_path = np.vstack((coef_path, coef))
            # coef_path = np.append(coef_path, coef).reshape((n_iter+1, nf))
            coef_path_copy = coef_path.copy()
            coef_path = np.zeros((coef_path_copy.shape[0]+1, coef_path_copy.shape[1]))
            coef_path[:-1] = coef_path_copy
            coef_path[-1] = coef
        if drop:
            nA -= 1            
            temp = Cov_copy[idx_drop] - G_copy[idx_drop] @ coef
            Cov[idx_drop] = temp
            sign_active[idx_drop] = 0
            idx_A_tf[idx_drop] = False
            idx_rest_tf[idx_drop] = True

    est_path = np.zeros_like(alpha_path)
    if best_estimator:
        for i in range(est_path.shape[0]):
            coef_i = coef_path[i]
            coef_i = np.ascontiguousarray(coef_i)
            k = i   # LARS
            # k = np.sum(coef_i!=0)
            if best_estimator == 'Cp':
                est_path[i] = np.sum((y - X @ coef_i)**2/(1.0/ynorm)**2) - ns + 2*k
            elif best_estimator == 'BIC':
                est_path[i] = np.sum((y - X @ coef_i)**2/(1.0/ynorm)**2) + np.log(ns) * k

        best_est_idx = np.argmin(est_path[1:])+1    # in some case the 1st fit was bad, best_est will choose the coefs=0 case
        coef = coef_path[best_est_idx]
        # estimator_path = est_path

    coef = np.around(coef, decimals=decimals)
    # Calculate the intercept and recover the coef & b for original X and y
    b = np.array(ymean - (Xmeans/Xnorms) @ coef * ynorm)
    coef = coef / Xnorms * ynorm
    if path:
        coef_path = coef_path / Xnorms[None,:] * ynorm
        coef_path = np.around(coef_path, decimals=decimals)

    return coef, b, coef_path, alpha_path, est_path

# jit accelerated erfcx for log_ndtr
@njit(fastmath=True)
def erfcx(x):
    if x < 25:
        return erfc(x) * np.exp(x*x)
    else:
        y = 1. / x
        z = y * y
        s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
        return s * 0.564189583547756287

# jit accelerated log_ndtr
@njit(fastmath=True)
def log_ndtr(x):
    t = x * msqrt1_2
    if x < -1:
        log_ndtr = np.log(erfcx(-t)/2) - t**2
    else:
        log_ndtr = np.log1p(-erfc(t)/2)
    return log_ndtr

@njit(fastmath=True)
def solve_lower_triangular(L, b):
    n = L.shape[0]
    x = np.empty_like(b)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * x[j]
        x[i] = s / L[i, i]
    return x

@njit(fastmath=True)
def probit_logcdf(u, cov, tol=1e-3, max_steps=200, e=0.01):
    ndim = cov.shape[0]
    I = np.eye(ndim)
    # if ndim = 1, use log_ndtr instead
    if ndim == 1:
        log_mep = log_ndtr(u[0]/cov[0][0]**0.5)
        max_change_rate = 0.0
        i_iters = 1
        convergence = True
    else:
        lmin = np.min(np.linalg.eigvalsh(cov))
        delta = e*lmin
        covt = cov - delta * I
        X = np.linalg.cholesky(covt)
        Omega = 1/delta * I
        # inv_Omega = delta * I
        # xi = scipy.linalg.solve_triangular(X, u/np.sqrt(delta), lower=True)
        xi = solve_lower_triangular(X, u/np.sqrt(delta))
        r0 = delta * xi
        V = 1/delta * X.T
        V = np.ascontiguousarray(V)
        logdet_Qep = ndim * np.log(delta)
        k = np.zeros(ndim)
        m = np.zeros(ndim)
        logZ = np.zeros(ndim)

        r_ep = r0
        # max_change_rate = np.inf
        i_iters = 0
        convergence = False
        # while (max_change_rate >= tol) and (i_iters <= max_steps):
        for j in range(max_steps):
            # max_change_rate = 0.0
            max_k_change_rate = 0.0
            max_m_change_rate = 0.0

            for i in range(ndim):
                # cavity distribution
                x_i = X[i,:]
                v_i = V[:,i].copy()
                xv = x_i @ v_i
                w_i = 1/(1 - k[i] * xv) * v_i
                r_minus_i = r_ep - m[i] * x_i

                # hybrid distribution
                xw = x_i @ w_i
                s_i = 1/np.sqrt(1 + xw)
                wr = w_i @ r_minus_i
                tau_i = s_i * wr

                # ith site approximation
                logPhi = log_ndtr(tau_i)
                logphi = (-0.5 * tau_i**2 - 0.5 * np.log(2.0*np.pi))
                zeta1 = np.exp(logphi - logPhi)
                zeta2 = -zeta1 * (tau_i + zeta1)

                ki_new = -zeta2 / (1 + xw + zeta2 * xw)
                ki_old = k[i]
                dki = ki_new - k[i]
                k[i] = ki_new

                # wr = w_i @ r_minus_i
                mi_new = zeta1*s_i + k[i]*wr + k[i]*zeta1*s_i*xw
                mi_old = m[i]
                dmi = mi_new - m[i]
                m[i] = mi_new

                denomZ = 1 + k[i]*xw
                numZ = 2*mi_new*wr + mi_new**2*xw - k[i]*wr**2
                logZ[i] = 0.5 * (numZ/denomZ - np.log(denomZ)) - logPhi

                # global approximation
                r_ep = r_minus_i + m[i] * x_i

                xV = x_i @ V    # TODO gives continuous warning
                denomV = 1 + dki * xv
                dV = -np.outer(v_i, (dki/denomV)*xV)
                V = V + dV
                logdet_Qep = logdet_Qep + np.log(denomV)

                if ki_old == 0:
                    max_k_change_rate = np.inf
                else:
                    max_k_change_rate = max(max_k_change_rate, np.fabs(dki/ki_old))

                if mi_old == 0:
                    max_m_change_rate = np.inf
                else:
                    max_m_change_rate = max(max_m_change_rate, np.fabs(dmi/mi_old))
            max_change_rate = max(max_k_change_rate, max_m_change_rate)
            i_iters = i_iters + 1

            if max_change_rate <= tol:
                convergence = True
                break

        # print(max_change_rate, i_iters)
        Omega_ep = Omega - 1/delta * (V @ (k[:,None]*X))    # TODO gives continuous warning
        # Omega_ep = 0.5 * (Omega_ep + Omega_ep.T)
        xi_ep = Omega_ep @ r_ep
        log_mep = 0.5 * (r_ep @ xi_ep - logdet_Qep - r0 @ xi + ndim*np.log(delta)) - np.sum(logZ)
    
    return log_mep, max_change_rate, i_iters, convergence

