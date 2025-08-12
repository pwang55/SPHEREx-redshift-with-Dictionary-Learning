## INSTALLATION

In desired local directoy,  
`$ git clone https://github.com/pwang55/SPHEREx-redshift-with-Dictionary-Learning.git`

## Update
```
$ git fetch
$ git status
$ git merge origin/main
```
or simply  
`$ git pull`

## Usage  
`python dictionary_learn.py config_default.yaml`

## Denpendencies on external files

A few training and validation sets are available on Onedrive `Pao_Yu/Catalogs/`:

### Training sets

`deepfield_7795_cut.npz` contains ~2k COSMOS deep-field level QuickCatalog sources (7795 sources with cosmology cut)  
`GAMA_cut_rand3000.npz` contains 3000 random selected GAMA QuickCatalog sources after cosmology cut  
`deepfield_7795_cut_GAMA_cut_rand3000.npz` combines the above two  

### Validation sets

`quickcat_110k_selected` contains ~20k COSMOS all-sky level QuickCatalog sources (110k with cosmology cut), older data. This is consistent with the deepfield set in the training catalogs.  
`COSMOS_110k_cut_quickcat_fiducial_parquet` contains the same sources as above but with the latest models  
`COSMOS_166k_cut_quickcat_fiducial_parquet` contains ~32k COSMOS all-sky QuickCatalog sources (166k with cosmology cut), with the latest models  
`GAMA_cut_quickcat_fiducial_parquet` contains ~42k GAMA all-sky QuickCatalog sources after cosmology cut using the latest models  

### Filters and central wavelengths

Up-to-date SPHEREx fiducial filters can be found on Onedrive `Pao_Yu/SPHEREx_filters/SPHEREx_102/`. These are actual measured SPHEREx fiducial filters, but cut away wavelength responses that are far from primary regions, for the sake of runtime efficiency. This results in up to 3% difference in responses.

`fiducial_filters.txt` contains the names of all filters. Required if using filter convolution.  
`fiducial_filters_cent_waves.txt` contains the central wavelengths of fiducial filters. This is an optional input for the config file.  

### Initial dictionaries

`brown_cosmos_kmean20.npz` is available on Onedrive `Pao_Yu/Initial Dictionaries/`. User has the option to use smaller than 20 dictionaries in config file.

