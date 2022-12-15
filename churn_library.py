""" Module that contains functions to find customers who are likely to churn

Author: Sameed Hayat
Date:   14th Dec 2022
"""


# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df_bank_data = pd.read_csv(pth)
        return df_bank_data
    except FileNotFoundError:
        print("Provided path doesn't exist")
        return None


def perform_eda(df_bank_data):
    '''
    perform eda on df and save figures to images folder
    input:
            df_bank_data: pandas dataframe

    output:
            None
    '''
    pth_for_images = "images/eda"
    df_bank_data_copy = df_bank_data.copy()
    # Churn Customer distribution
    df_bank_data_copy['Churn'] = df_bank_data_copy['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df_bank_data_copy['Churn'].hist()
    plt.savefig(os.path.join(pth_for_images, 'churn_hist.png'))

    # Customer age distribution
    plt.figure(figsize=(20, 10))
    df_bank_data_copy['Customer_Age'].hist()
    plt.savefig(os.path.join(pth_for_images, 'customer_age_hist.png'))

    # Marital Status distribution
    plt.figure(figsize=(20, 10))
    df_bank_data_copy.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(pth_for_images, 'marital_status_hist.png'))

    # Density plot for Total_Trans_Ct
    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df_bank_data_copy['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(pth_for_images, 'density_total_trans.png'))

    # Correlation plot
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df_bank_data_copy.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(os.path.join(pth_for_images, 'heatmap.png'))
    plt.close()
    return df_bank_data_copy


def encoder_helper(df_bank_data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_bank_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    df_bank_data_copy = df_bank_data.copy()

    if not response:
        response = 'Churn'

    # gender encoded column
    for column in category_lst:
        column_lst = []
        column_groups = df_bank_data.groupby(column).mean()['Churn']

        for val in df_bank_data[column]:
            column_lst.append(column_groups.loc[val])

        df_bank_data_copy['{}_{}'.format(column, response)] = column_lst
    return df_bank_data_copy


def perform_feature_engineering(df_bank_data, response):
    '''
    input:
              df_bank_data: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    if not response:
        response = 'Churn'

    # Categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # Columns to keep used for analysis
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_' + response,
        'Education_Level_' + response,
        'Marital_Status_' + response,
        'Income_Category_' + response,
        'Card_Category_' + response]

    encoded_df = encoder_helper(df_bank_data, cat_columns, response)

    # Training Data keeping the columns required for analysis
    x_data = pd.DataFrame()
    x_data = encoded_df[keep_cols]

    # Labeled Data
    y_data = encoded_df['Churn']

    # Train Test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    return (x_train, x_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    pth_for_images = 'images/results'

    # Plotting classification report for Random Forest model
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(os.path.join(pth_for_images, 'rf_classification_report.png'))

    # Plotting classification report for Logistic regression model
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(os.path.join(pth_for_images, 'lr_classification_report.png'))


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth, 'feature_importance_plot.png'))


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Random Forest model training
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    # Random Forest model prediction
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # Logistic Regression model training
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    # Logistic Regression model prediction
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # plotting roc curve
    plt.figure(figsize=(15, 8))
    axis_plt = plt.gca()

    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis_plt,
        alpha=0.8)
    
    lrc_plot = plot_roc_curve(lrc, x_test, y_test, ax=axis_plt, alpha=0.8)
    

    plt.savefig('images/results/roc_curve.png')
    plt.close()

    # saving classification plot image
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # saving feature importance plots
    feature_importance_plot(cv_rfc, x_train, 'images/results/')


if __name__ == "__main__":
    PTH_TO_DATA = 'data/bank_data.csv'

    # Import bank data
    DF = import_data(PTH_TO_DATA)

    # Performing exploratory data analysis on dataframe
    DF = perform_eda(DF)

    # Performing feature engineering
    (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST) = perform_feature_engineering(DF, 'Churn')

    # Training all models and saving results
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
