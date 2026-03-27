#IMPORTS
import joblib
import os
import sklearn.linear_model as skl_lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

from feature_engineering.build_features import LogisticRegressionFeatureEngineering
from models.predict_model import BasicLogisticRegressionPredictor


class BasicLogisticRegressionTraining:
    def __init__ (self, train_df, test_df):
        """
        Initialize the logistic regression model
        
        Args:
            train_path: Path to training dataframe
            test_path: Path to testing dataframe
        """
        self.train_df = train_df
        self.test_df = test_df
        self.scaler = StandardScaler()
        self.feature_engineer = LogisticRegressionFeatureEngineering()
        #Placeholders
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
        #Model
        self.model = skl_lm.LogisticRegression(solver = "newton-cg",max_iter=1000,class_weight="balanced")
    
    def set_features(self):
        """
        Takes the set of features and target variable from
        the "LogisticRegressionFeatureEngineering" class
        """
        features = self.feature_engineer.feature_cols
        target_variable = self.feature_engineer.target_col
        
        self.X_train = self.train_df[features]
        self.y_train = self.train_df[target_variable]
        self.X_test = self.test_df[features]
        self.y_test = self.test_df[target_variable]
    
    def train(self):
        """
        Train the logistic regression model.
        The scaler standardizes the features to fit the requirements of
        logistic regression.
        The model and scaler are then saved so they can be called upon later without the need
        for retraining.
        """
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.model.fit(self.X_train, self.y_train)
        
        os.makedirs("models", exist_ok=True)
        
        joblib.dump(self.model, 'models/logistic_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
    def evaluate(self):
        """
        Creates a predictor object from "BasicLogisticRegressionPredictor".
        Predictions are then generated and metrics accuracy, precision,
        recall, f1_score, and a confusion matrix are printed to the 
        console.

        Returns:
            metrics_dict: A dictionary of the resulting metrics. 
        """
        predictor = BasicLogisticRegressionPredictor()
        
        self.y_pred, _ = predictor.predict(self.X_test)

        metrics_dict = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred),
            "f1_score": f1_score(self.y_test, self.y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred)
        }

        return metrics_dict
    
    def run_lr_pipeline(self):
        self.set_features()
        self.train()
        return self.evaluate()