#IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc

class LogisticRegressionVisualizer():
    def __init__(self, model, feature_cols):
        self.model = model
        self.features = feature_cols
        
        
    def plot_feature_importance(self):
        """
        Create a horizontal bar plot showing the weight of 
        the coefficient for each feature.
        """
        #Gather feature coefficients
        coefficients = self.model.coef_[0]
        #Create a dataframe of model features and their coefficients
        feature_importance = pd.DataFrame({
            "Feature": self.features,
            "Coefficient": coefficients
        }).sort_values(by="Coefficient", key=np.abs)
        
        #Create plot
        plt.figure(figsize=(13,7))
        sns.barplot(data=feature_importance, x="Coefficient", y="Feature",
                    hue=np.where(feature_importance["Coefficient"] <0, "Negative", "Positive"),
                    palette = {"Negative": "tomato", "Positive": "steelblue"})
        plt.xlabel("Strength of coefficient β")
        plt.ylabel("Name of feature")
        plt.tight_layout()
        #Save resulting plot to /reports/figures
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{base_dir}/../../reports/figures", exist_ok=True)
        plt.savefig(f"{base_dir}/../../reports/figures/feature_importance.png")
        plt.show()

class RandomForestVisualizer():
    def __init__(self, model, feature_cols):
        self.model = model
        self.features = feature_cols

    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            "Feature": self.features,
            "Importance": importances
        }).sort_values(by="Importance")

        plt.figure(figsize=(13, 7))
        sns.barplot(data=feature_importance, x="Importance", y="Feature", color="steelblue")
        plt.xlabel("Feature Importance (Gini)")
        plt.ylabel("Name of feature")
        plt.tight_layout()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{base_dir}/../../reports/figures", exist_ok=True)
        plt.savefig(f"{base_dir}/../../reports/figures/rf_feature_importance.png")
        plt.show()

class TFIDFVisualizer():
    def __init__(self, model, feature_cols):
        self.model = model
        self.features = feature_cols

    def plot_feature_importance(self, top_n=20):
        coefficients = self.model.coef_[0]
        feature_importance = pd.DataFrame({
            "Feature": self.features,
            "Coefficient": coefficients
        })
        feature_importance = feature_importance.reindex(
            feature_importance["Coefficient"].abs().nlargest(top_n).index
        ).sort_values(by="Coefficient", key=np.abs)

        plt.figure(figsize=(13, 7))
        sns.barplot(data=feature_importance, x="Coefficient", y="Feature",
                    hue=np.where(feature_importance["Coefficient"] < 0, "Negative", "Positive"),
                    palette={"Negative": "tomato", "Positive": "steelblue"})
        plt.xlabel("Strength of coefficient β")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} TF-IDF Feature Importances")
        plt.tight_layout()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{base_dir}/../../reports/figures", exist_ok=True)
        plt.savefig(f"{base_dir}/../../reports/figures/tfidf_feature_importance.png")
        plt.show()
        
        
class LSTMVisualizer():
    def __init__(self, model, history, y_test, y_pred, y_pred_proba):
        self.model = model
        self.history = history
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
    def plot_training_history(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.history.history["loss"], label="Training Loss", color="steelblue")
        plt.plot(self.history.history["val_loss"], label = "Validation Loss", color = "tomato")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        os.makedirs(f"{self.base_dir}/../../reports/figures", exist_ok=True)
        plt.savefig(f"{self.base_dir}/../../reports/figures/LSTM_training_history.png")
        plt.show()
        
    def plot_confusion_matrix(self):
        """
        Plots a heatmap of the confusion matrix showing true vs predicted labels.
        """
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actually Negative', 'Actually Positive'])
        plt.title("Confusion Matrix")
        plt.tight_layout()
        os.makedirs(f"{self.base_dir}/../../reports/figures", exist_ok=True)
        plt.savefig(f"{self.base_dir}/../../reports/figures/LSTM_confusion_matrix_plot.png")
        plt.show()
        
    def plot_roc_curve(self):
        """
        Plots the ROC curve and displays the AUC score.
        Summarizes model performance across all classification thresholds.
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='steelblue', label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='tomato', linestyle='--', label='Random Classifier')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        os.makedirs(f"{self.base_dir}/../../reports/figures", exist_ok=True)
        plt.savefig(f"{self.base_dir}/../../reports/figures/LSTM_roc_curve.png")
        plt.show()

    def plot_all(self):
        """
        Run all visualizations
        """
        print(f"history: {self.history}")
        print(f"y_pred: {self.y_pred}")
        print(f"y_pred_proba: {self.y_pred_proba:}")
        self.plot_training_history()
        self.plot_confusion_matrix()
        self.plot_roc_curve()