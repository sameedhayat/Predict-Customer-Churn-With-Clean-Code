# Predict-Customer-Churn-With-Clean-Code

## Project description
Identifyinf credit card customers that are most likely to churn. The project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package also have the flexibility of being run interactively or from the command-line interface (CLI).

The code in churn_library.py should complete data science solution process including:

* EDA
* Feature Engineering (including encoding of categorical variables)
* Model Training
* Prediction
* Model Evaluation

The sequence diagram below shows the sequence of churn_library.py function calls, and the Document Strings section further shows each function's input/output details.
![Sequence Diagram](images/sequence_diagram.jpeg)


## Files and data description
This project follows the following directory structure

* /data: Input data file is available in the data directory as bank_data.csv
* /images: contains images for both exploratory data analysis plots and results plots
* /logs: contains the logs for the testing
* /models: contains pkl files for models

## Running the files 

### Installing all the dependencies

1. Run pip command on the terminal install all the dependencies for python3.6 or python3.8

For python 3.6 run the following command

```bash
pip install -r requirements_py3.6.txt
```

For python 3.8 run the following command

```bash
pip install -r requirements_py3.8.txt
```

### Training the model and saving all the results

1. After installing all the dependencies run the code in the churn library, which saves all the eda plots, results plots and models

```bash
python churn_library.py
```

2. For testing churn library run the code in the churn_script_logging_and_tests.py

```bash
python churn_script_logging_and_tests.py
```

3.  All the code written for this project follows the PEP 8 guidelines.
Run the following commands to test

```bash
pylint churn_library.py
```

```bash
pylint churn_script_logging_and_tests.py
```


