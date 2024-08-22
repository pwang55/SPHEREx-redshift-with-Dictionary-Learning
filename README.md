## INSTALLATION

In desired local directoy,

`$ git clone https://github.com/pwang55/SPHEREx-redshift-with-Dictionary-Learning.git`

## Update

`$ git fetch`

`$ git status`

`$ git merge origin/main`

or simply

`$ git pull`

##

Make sure **`main_dictionary.py`** and **`dictionary_learn_fx.py`** are in the same folder.

By default, `main_dictionary.py` and `dictionary_learn_from_config.py` creates an `OUTPUT` folder for all output files.

## Usage

`$ python main_dictionary.py input_file.npz`

Or

`$ python dictionary_learn_from_config.py config.yaml`

Required files:
 - 7 EAZY templates (configure folder location in main_dictionary.py: `eazy_templates_location`)
 - SPHEREx filters (configure folder locaiton in main_dictionary.py: `filter_location`)



