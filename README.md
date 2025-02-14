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

##

Make sure **`dictionary_learn_from_config.py`** and **`dictionary_learn_fx.py`** are in the same folder.  
By default, `dictionary_learn_from_config.py` creates an `OUTPUT` folder for all output files.

## Usage  
`$ python dictionary_learn_from_config.py config_default.yaml`

Required files:
 - 7 EAZY templates (configure folder location in config_default.yaml: `eazy_templates_location`)
 - SPHEREx filters (configure folder locaiton in config_default.yaml: `filter_location`)



