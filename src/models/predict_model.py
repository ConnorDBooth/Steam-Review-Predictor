
#IMPORTS
import joblib


class BasicLogisticRegressionPredictor:
    def __init__(self, model_path='models/logistic_model.pkl', scaler_path='models/scaler.pkl'):
        """
        Loads a pre-trained logistic regression model.
        Args:
            model_path (str, optional): Defaults to 'models/logistic_model.pkl'.
            scaler_path (str, optional): Defaults to 'models/scaler.pkl'.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
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
    