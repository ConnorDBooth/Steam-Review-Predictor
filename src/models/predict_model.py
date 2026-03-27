
#IMPORTS
import joblib


class BasicLogisticRegressionPredictor:
    def __init__(self, model_path='models/logistic_model.pkl', scaler_path='models/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        return y_pred, y_pred_proba
    