#IMPORTS
import joblib
import os
import sklearn.linear_model as skl_lm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

from feature_engineering.build_features import LogisticRegressionFeatureEngineering
from models.predict_model import BasicLogisticRegressionPredictor, BasicRandomForestPredictor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.utils.class_weight import compute_class_weight
from feature_engineering.build_features import LSTMFeatureEngineering


class BasicLogisticRegressionTraining:
    def __init__ (self, train_df, test_df, feature_engineer=None):
        """
        Initialize the logistic regression model
        
        Args:
            train_path: Path to training dataframe
            test_path: Path to testing dataframe
        """
        self.train_df = train_df
        self.test_df = test_df
        self.scaler = StandardScaler()
        #Placeholders
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.feature_engineer = feature_engineer
        
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
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        joblib.dump(self.model, f'{base_dir}/logistic_model.pkl')
        joblib.dump(self.scaler, f'{base_dir}/scaler.pkl')
        
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

class BasicRandomForestTraining:
    def __init__(self, train_df, test_df, featureEng=None):
        """
        Initialize the Random Forest model.

        Args:
            train_df: Training dataframe
            test_df: Testing dataframe
        """
        self.train_df = train_df
        self.test_df = test_df
        self.feature_engineer = featureEng or LogisticRegressionFeatureEngineering()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        )

    def set_features(self):
        """
        Takes the set of features and target variable from
        the "LogisticRegressionFeatureEngineering" class.
        """
        features = self.feature_engineer.feature_cols
        target_variable = self.feature_engineer.target_col

        self.X_train = self.train_df[features]
        self.y_train = self.train_df[target_variable]
        self.X_test = self.test_df[features]
        self.y_test = self.test_df[target_variable]

    def tune_hyperparameters(self):
        """
        Usee RandomizedSearchCV to find optimal hyperparameters for
        RandomForest model

        Returns:
            best_params_: Dictionary of the best hyperparams
        """
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"]
        }
        
        random_search = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=42),
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring="f1",
            verbose=1,
            random_state=42
        )

        random_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best F1 score: {random_search.best_score_:.4f}")

        return random_search.best_params_
    
    def train(self, best_params=None):
        """
        Trains the Random Forest model and saves it to disk.
        No scaling required since decision trees are not
        sensitive to feature scale.
        """
        if best_params:
            self.model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            **best_params
        )
        self.model.fit(self.X_train, self.y_train)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        joblib.dump(self.model, f"{base_dir}/random_forest_model.pkl")

    def evaluate(self):
        """
        Generates predictions and returns evaluation metrics.

        Returns:
            metrics_dict: A dictionary of the resulting metrics.
        """
        predictor = BasicRandomForestPredictor()
        self.y_pred, _ = predictor.predict(self.X_test)
        metrics_dict = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred),
            "f1_score": f1_score(self.y_test, self.y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred)
        }
        return metrics_dict

    def run_rf_pipeline(self):
        self.set_features()
        self.train()
        #Train without tuning to compare results
        default_results = self.evaluate()
        print("\nDefault Random Forest Results:")
        for k, v in default_results.items():
            print(f"{k}: {v}")
        #Train with tuning
        best_params = self.tune_hyperparameters()
        self.train(best_params)
        return self.evaluate()
    
class LSTMTraining:
    def __init__(self, train_df, test_df):
        """
        Initialize the LSTM model.

        Args:
            train_df: training dataframe, taken from LSTMFeatureEngineering
            test_df: testing dataframe, taken from LSTMFeatureEngineering
        """
        self.train_df = train_df
        self.test_df = test_df
        self.feature_engineer = LSTMFeatureEngineering()
        self.embedding_dim = 50 #The size of the vector used to represent each word
        
        # Size of the hidden state vector. 
        # A higher value means the model can remember more complex patterns 
        self.lstm_units = 32 
        #Placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Builds and compiles LSTM model.
        Embedding layer learns word representations,
        LSTM layer captures sequential context,
        Dense layer with sigmoid activation is for binary classification
        """
        
        self.model = Sequential([
            Embedding(
                input_dim=self.feature_engineer.max_words,
                output_dim = self.embedding_dim,
                input_length= self.feature_engineer.max_len
            ),
            LSTM(units=self.lstm_units),
            Dense(1, activation="sigmoid")
        ])
        
        self.model.compile(
            loss="binary_crossentropy",
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
        
    def set_features(self):
        """
        Runs tokenization and padding on the training and test set
        by calling fit_transfrom from LSTMFeatureEngineering.
        Populates X_train, X_test, y_train, y_test
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineer.fit_transform(self.train_df, self.test_df)
        
    def train(self):
        """
        Computes balanced class weights to account for positve/negative
        review imbalance. Penalizes the model more for misclassifying negatives.
        Then trains and saves the model
        """
        class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weight))
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs = 10,
            batch_size = 32,
            validation_split = 0.1,
            class_weight=class_weight_dict
        )
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model.save(os.path.join(base_dir, "lstm_model.keras"))
    
    def tune_hyperparameters(self, n_iter=10):
        """
        Performs random search over LSTM hyperparameters.
        Trains n_iter random combinations and keeps the best by F1 score.

        Args:
            n_iter: Number of random combinations to try. Defaults to 10.

        Returns:
            best_params: Dictionary of the best hyperparameters found.
        """
        import random

        param_distributions = {
            "lstm_units":    [16, 32, 64, 128],
            "embedding_dim": [32, 50, 100],
            "epochs":        [5, 10],
            "batch_size":    [32, 64, 128]
        }

        class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weight))

        results = []

        for i in range(n_iter):
            params = {k: random.choice(v) for k, v in param_distributions.items()}
            print(f"\nTrial {i+1}/{n_iter}: {params}")

            model = Sequential([
                Embedding(
                    input_dim=self.feature_engineer.max_words,
                    output_dim=params["embedding_dim"],
                    input_length=self.feature_engineer.max_len
                ),
                LSTM(units=params["lstm_units"]),
                Dense(1, activation="sigmoid")
            ])
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

            model.fit(
                self.X_train, self.y_train,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                validation_split=0.1,
                class_weight=class_weight_dict,
                verbose=0
            )

            y_pred = (model.predict(self.X_test) > 0.5).astype(int).flatten()
            score = f1_score(self.y_test, y_pred)
            print(f"  F1: {score:.4f}")

            results.append((score, params, model))

        results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_params, best_model = results[0]

        print(f"\nBest parameters: {best_params}")
        print(f"Best F1: {best_score:.4f}")

        # Replace model with best found
        self.model = best_model
        self.embedding_dim = best_params["embedding_dim"]
        self.lstm_units = best_params["lstm_units"]

        return best_params

    def evaluate(self):
        """
        Generates predictions and returns evaluation metrics.

        Returns:
            metrics_dict: A dictionary of the resulting metrics.
        """

        y_pred_proba = self.model.predict(self.X_test)
        self.y_pred_proba = y_pred_proba
        self.y_pred = (y_pred_proba > 0.5).astype(int).flatten() 
        
        metrics_dict = {
            "loss": self.model.evaluate(self.X_test, self.y_test, verbose=0)[0],
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred),
            "f1_score": f1_score(self.y_test, self.y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred)
        }
        return metrics_dict
    
    def run_lstm_pipeline(self, tune=False, n_iter=10):
        self.set_features()
        self.build_model()
        self.train()

        # Baseline results
        default_results = self.evaluate()
        print("\nDefault LSTM Results:")
        for k, v in default_results.items():
            print(f"{k}: {v}")

        if tune:
            best_params = self.tune_hyperparameters(n_iter=n_iter)
            self.train()
            return self.evaluate()

        return default_results