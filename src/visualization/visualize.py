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