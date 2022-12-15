""" Test file for churn library

Author: Sameed Hayat
Date:   14th Dec 2022
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_bank_data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df_bank_data.shape[0] > 0
        assert df_bank_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''

    df_bank_data = cls.import_data("./data/bank_data.csv")
    pth_for_images = "images/eda"

    try:
        perform_eda(df_bank_data)
        logging.info("Testing perform_eda: SUCCESS")
    except BaseException:
        logging.error("Testing perform_eda: FAILED")
        raise

    try:
        assert os.path.exists(os.path.join(pth_for_images, 'churn_hist.png'))
        logging.info(
            "Testing perform_eda: Churn Histogram file found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Churn Histogram file not found : FAILED")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                pth_for_images,
                'customer_age_hist.png'))
        logging.info(
            "Testing perform_eda: Customer Age Histogram file found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Customer Age Histogram file not found : FAILED")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                pth_for_images,
                'marital_status_hist.png'))
        logging.info(
            "Testing perform_eda: Marital Status Histogram file found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda:  Marital Status file not found : FAILED")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                pth_for_images,
                'density_total_trans.png'))
        logging.info("Testing perform_eda: Density Total file found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Density Total file not found : FAILED")
        raise err

    try:
        assert os.path.exists(os.path.join(pth_for_images, 'heatmap.png'))
        logging.info("Testing perform_eda: Heatmap file found : SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: Heatmap file not found : FAILED")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df_bank_data = cls.import_data("./data/bank_data.csv")
    df_bank_data = cls.perform_eda(df_bank_data)

    # Categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # With Categorical Columns
    try:
        encoded_df = encoder_helper(df_bank_data, cat_columns, 'Churn')
        logging.info("Testing encoder_helper: SUCCESS")
    except BaseException:
        logging.error("Testing encoder_helper: FAILED")
        raise

    try:
        assert df_bank_data.equals(encoded_df) == False
        logging.info(
            "Testing encoder_helper: With categorical Column dataframes are not same: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Without categorical Column dataframes are same: FAILED")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df_bank_data = cls.import_data("./data/bank_data.csv")
    df_bank_data = cls.perform_eda(df_bank_data)

    try:
        (x_train, x_test, y_train, y_test) = perform_feature_engineering(
            df_bank_data, 'Churn')
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except BaseException:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise

    # Checking if after split sum of train and test rows are equal to
    # dataframe rows
    try:
        assert (x_train.shape[0] + x_test.shape[0]) == df_bank_data.shape[0]
        logging.info(
            "Testing perform_feature_engineering: Rows of the train \
			and test data matches the whole data: SUCCESS")
    except AssertionError as err:
        logging.info(
            "Testing perform_feature_engineering: Rows of the train \
				 and test data doesn't match the whole data: SUCCESS")
        raise err

    # Checking if after split sum of train and test labels are equal to
    # dataframe rows
    try:
        assert (y_train.shape[0] + y_test.shape[0]) == df_bank_data.shape[0]
        logging.info(
            "Testing perform_feature_engineering: Rows of the train \
			and test labels matches the whole data: SUCCESS")
    except AssertionError as err:
        logging.info(
            "Testing perform_feature_engineering: Rows of the train \
				 and test labels doesn't match the whole data: SUCCESS")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''

    pth_for_models = './models'
    pth_for_reports = './images/results'

    df_bank_data = cls.import_data("./data/bank_data.csv")
    df_bank_data = cls.perform_eda(df_bank_data)
    (x_train, x_test, y_train, y_test) = cls.perform_feature_engineering(
        df_bank_data, 'Churn')

    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except BaseException:
        logging.error("Testing train_models: FAILED")

    # Testing for logistic regression model
    try:
        assert os.path.exists(
            os.path.join(
                pth_for_models,
                'logistic_model.pkl'))
        logging.info(
            "Testing train_models: logistic model file found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: logistic model not found : FAILED")
        raise err

    # Testing for Random Forest model
    try:
        assert os.path.exists(os.path.join(pth_for_models, 'rfc_model.pkl'))
        logging.info(
            "Testing train_models: Random forest model file found : SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: Random forest not found : FAILED")
        raise err

    # Testing for Feature importance file
    try:
        assert os.path.exists(
            os.path.join(
                pth_for_reports,
                'feature_importance_plot.png'))
        logging.info(
            "Testing train_models: Feature importance file found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Feature importance file not found : FAILED")
        raise err

    # Testing for Linear Regression Classification Report file
    try:
        assert os.path.exists(
            os.path.join(
                pth_for_reports,
                'lr_classification_report.png'))
        logging.info(
            "Testing train_models: Linear Regression Classification Report found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Linear Regression Classification Report not found : FAILED")
        raise err

    # Testing for Random Forest Classification Report file
    try:
        assert os.path.exists(
            os.path.join(
                pth_for_reports,
                'rf_classification_report.png'))
        logging.info(
            "Testing train_models: Random Forest Classification Report found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Random Forest Classification Report not found : FAILED")
        raise err

    # Testing for ROC Curve file
    try:
        assert os.path.exists(os.path.join(pth_for_reports, 'roc_curve.png'))
        logging.info("Testing train_models: ROC Curve File found : SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: ROC Curve File not found : FAILED")
        raise err


if __name__ == "__main__":
    # Testing all the churn library functions
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
