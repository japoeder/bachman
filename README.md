# Bachman

### Introduction

Stream simulator for dev and model / scenario testing.

1. Step 1.

### Running the model

1. Local
2. Download and install Python on your computer
3. Clone this repo, and navigate to [pid_0001_collectability_model](.) directory (containing the README.md)
4. Set up environment by running the following in the terminal:

   1. conda create --name pid_0001_collectability_model python=3.10
   2. conda activate pid_0001_collectability_model
   3. conda config --add channels conda-forge
   4. pip install -r requirements.txt
5. To run the model, run the Flask app and submit a POST request in the proper form
6. There's a sample POST request in 'collectability_model/templates', as well as samples to illustrate what data is required for base and enhnced models

The result is a response object in JSON with the model results.  Note that the model inputs and results are also output to S3 in the following buckets:

* s3://collectability-mlops/json-requests/
* s3://collectability-mlops/json-responses/
* s3://collectability-mlops/html-responses/

### Model Details:

```commandline
usage: POST request
  
optional "parameters "arguments:
  cos_name_val	  		cos similarity score for name validation. 'y' (default) or 'n'
  tuning_file	  		file with tuning sensitivities. collectability/tuning.json used by default.
  version            		model version. 'base' (default) or 'enhanced'.
```

### Request Structure

header:

* key / value -> "X-Amzn-SageMaker-Inference-Component":"Model-1715285187796-20240509-2034510"

body:

{

    "authentication":{

    ...

    },"parameters":{

    ...

    }, "scenario_data":{

    ...

    }, "enhanced_model_elements":{

    ...

    }

}

## Python Packaging

This code is packaged as a python module, with the structure outlined in the section below.

### Project Layout

* [pid_0001_collectability_model_v1](collectability_model): The parent or "root" folder containing all of these files. Can technically have any name.
  * [README.md](README.md):
    The guide you're reading.
  * [`__init__.py`](lab1/__init__.py)
    For package / module structure.
  * [`.`](lab1/__init__.py)gitignore
    Version control doc
  * .pre-commit-config.yaml
    Pre-commit hooks for formatting and linting
  * .pylintrc
    Linter parameter file.
  * [`lint.py`](lab1/__init__.py)
    Linter driver.
  * [`p`](lab1/__init__.py)yproject.toml
    Black exception logic.
  * [`r`](lab1/__init__.py)eq.txt
    Required libraries for model.
  * [collectability_model](.): This is the *module* in our *package*.
    * [`html`](lab1/__init__.py)
      Directory that holds html output
    * [`t`](lab1/__init__.py)emplates
      Directory that holds templates for base and enhanced model scenarios and tuning.
    * [`u`](lab1/__init__.py)tilities
      * [`m`](lab1/__init__.py)ake_json.py
        Shell for a program to create a json payload if data comes in another format
      * [`m`](lab1/__init__.py)l_utils.py
        Cos similarity model
      * open_html.py
        Opens html output if requested
    * [`__init__.py`](lab1/__init__.py)
      Expose what functions, variables, classes, etc when scripts import this module.
    * [`__main__.py`](lab1/__main__.py)
      This file is the entrypoint to your program when ran as a program.
    * `collectability_model.py`
      Driver program for the module
    * `build_report_template.py`
      Customizes html template for score report
    * `load_scenario.py`
      Method to read in data
    * `load_tuning.py`
      Helper method to read in tuning parameters
    * `proc_wams.py`
      Create weights and measures used in methodology
    * `score_report.py`
      Generate the final score report using customized template
    * `stage_i_scoring.py`
      Stage I scoring to assess fraud
    * `stage_ii_scoring.py`
      Stage II scoring to get modeled collectability score
    * `stage_iii_planning_calcs.py`
      Run final calculations for payment / settlement planning
    * [`t`](lab1/__init__.py)uning.json
      Default tuning sensitivities for attribute weights
    * `validate_input_files.py`
      Validates the json inputs (raw data and tuning)
    * `utilities/make_json.py`
      Use to format raw data if not in json format
    * `utilities/open_html.py`
      Opens the system browser and displays report
