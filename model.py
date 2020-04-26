import sys
import os
import numpy as np
import pandas as pd
import warnings
import dill as pickle

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib


class PreProcessing(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, df):
        pred_var = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome",
                    "CoapplicantIncome", "LoanAmount",
                    "Loan_Amount_Term", "Credit_History", "Property_Area"]
        df = df[pred_var]
        df['Dependents'] = df['Dependents'].fillna(0)
        df['Married'] = df['Married'].fillna('No')
        df['Gender'] = df['Gender'].fillna('Male')
        df['Self_Employed'] = df['Self_Employed'].fillna('No')
        df['LoanAmount'] = df['LoanAmount'].fillna(self.term_mean_)
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(self.amt_mean_)
        df['Credit_History'] = df['Credit_History'].fillna(1)

        gender_values = {'Female': 0, 'Male': 1}
        married_values = {'No': 0, 'Yes': 1}
        education_values = {'Graduate': 0, 'Not Graduate': 1}
        employed_values = {'No': 0, 'Yes': 1}
        property_values = {'Rural': 0, 'Urban': 1, 'Semiurban': 2}
        dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}

        df.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values,
                    'Self_Employed': employed_values, 'Property_Area': property_values, 'Dependents': dependent_values},
                   inplace=True)

        return df.values

    def fit(self, df, y=None, **fit_params):
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self


def build_and_train():
    data = pd.read_csv("loan_predictions/train.csv")
    pred_var = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
                "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"]
    X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'], test_size=0.25,
                                                        random_state=42)

    preprocess = PreProcessing()
    preprocess.fit(X_train)
    X_train_transformed = preprocess.transform(X_train)
    X_test_transformed = preprocess.transform(X_test)
    y_test = y_test.replace({'Y': 1, 'N': 0}).values
    y_train = y_train.replace({'Y': 1, 'N': 0}).values
    param_grid = {"randomforestclassifier__n_estimators": [10, 20, 30],
                  "randomforestclassifier__max_depth": [None, 6, 8, 10],
                  "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20],
                  "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}
    pipe = make_pipeline(PreProcessing(),
                         RandomForestClassifier())
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    grid.fit(X_train, y_train)
    print("Best parameters: {}".format(grid.best_params_))
    print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
    return (grid)


if __name__ == '__main__':
    model = build_and_train()

    filename = 'model_v1.pk'
    with open('loan_predictions/' + filename, 'wb') as file:
        pickle.dump(model, file)
