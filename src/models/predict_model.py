
#IMPORTS
import joblib
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from feature_engineering.build_features import LSTMFeatureEngineering


class BasicLogisticRegressionPredictor:
    def __init__(self):
        """
        Loads a pre-trained logistic regression model.
        Args:
            model_path (str, optional): Defaults to 'models/logistic_model.pkl'.
            scaler_path (str, optional): Defaults to 'models/scaler.pkl'.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = joblib.load(f"{base_dir}/logistic_model.pkl")
        self.scaler = joblib.load(f"{base_dir}/scaler.pkl")
        
    def predict(self, X):
        """
        Scales the features using the loaded scaler.
        Makes predictions based on the testing data.

        Args:
            X: features of model

        Returns:
            y_pred: Hard predictions
            y_pred_proba: The confidence
        """
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        return y_pred, y_pred_proba
    
class BasicRandomForestPredictor:
    def __init__(self):
        """
        Loads a pre-trained Random Forest model.
        Args:
            model_path (str, optional): Defaults to 'models/random_forest_model.pkl'.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = joblib.load(f"{base_dir}/random_forest_model.pkl")

    def predict(self, X):
        """
        Makes predictions based on the testing data.
        No scaling required since Random Forest is not sensitive to feature scale.
        
        Args:
            X: Features of model
        Returns:
            y_pred: Hard predictions
            y_pred_proba: The confidence
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        return y_pred, y_pred_proba
    
    class LSTMPredictor:
        def __init__(self):
            """
            Loads the trained LSTM model and tokenizer from the models directory.
            """
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model = load_model(os.path.join(base_dir, "lstm_model.keras"))
            self.tokenizer = joblib.load(os.path.join(base_dir, "lstm_tokenizer.pkl"))
            self.max_length = LSTMFeatureEngineering.max_len

        def predict(self, texts):
            """
            Takes a list of raw review strings and returns predictions.
            Args:
                texts: A list of review strings
            Returns:
                y_pred: Hard predictions (0 or 1)
                y_pred_proba: Confidence scores between 0 and 1
            """
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')

            y_pred_proba = self.model.predict(padded).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            return y_pred, y_pred_proba
