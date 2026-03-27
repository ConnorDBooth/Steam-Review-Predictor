
#IMPORTS
import joblib
import os


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
    