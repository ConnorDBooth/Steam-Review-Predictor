#IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class LogisticRegressionVisualizer():
    def __init__(self, model, feature_cols):
        self.model = model
        self.features = feature_cols
        
        
    def plot_feature_importance(self):
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