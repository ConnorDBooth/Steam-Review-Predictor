#IMPORTS
import sklearn.linear_model as skl_lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

#Need to move the prediction part to src/models/predict_model.py
class BasicLogisticRegressionTraining:
    def __init__ (self, train_path, test_path):
        """
        Initialize the logistic regression model
        
        Args:
            train_path: Path to training dataset
            test_path: Path to testing dataset
        """
        self.train_path = train_path
        self.test_path = test_path
        
        #Placeholders
        
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        #Model
        self.model = skl_lm.LogisticRegression(solver = "newton-cg",max_iter=1000,class_weight="balanced")
        
        #Scaler
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Loads and sets the training and testing dataframes
        
        Returns:
            Training and testing dataframes
        """
        try:
            self.train_df = pd.read_csv(self.train_path)
            self.test_df = pd.read_csv(self.test_path)
            return self.train_df, self.test_df
        except FileNotFoundError as e :
            print(f"Error: could not load data: {e}")
            return None, None
        
        
    def prepare_features(self):
        """
        Separate train and test data into features and target variable.
        
        Use all given features now. Once exploration has been done,
        features can be removed.
        
        Something to think about is the features that come after a review is posted
        vs when the review is originally posted.
        
        Ex. "votes_up" could be a good indicator of a positive review,
        but not until many people have seen the review. 
        
        Returns:
            self.X_train, self.y_train, self.X_test, self.y_test
        """
        if self.train_df is None or self.test_df is None:
            print("Error: Train and test data must be loaded first")
            return None, None, None, None
        target_col = "voted_up"
        if target_col not in self.train_df.columns or target_col not in self.test_df.columns:
            print("Error: target column not found")
            return None, None, None, None
        
        #Remove target variable and "review" from feature list since "review" is language based.
        feature_cols = [
            "author_steamid",
            "author_num_games_owned",
            "author_num_reviews",
            "author_playtime_forever",
            "author_playtime_last_two_weeks",
            "author_playtime_at_review",
            "votes_up",
            "votes_funny",
            "weighted_vote_score",
            "comment_count",
            "steam_purchase",
            "received_for_free",
            "written_during_early_access",
        ]
        
        self.X_train = self.train_df[feature_cols]
        self.y_train = self.train_df[target_col]
        
        self.X_test = self.test_df[feature_cols]
        self.y_test = self.test_df[target_col]
        
        
        return self.X_train, self.y_train, self.X_test, self.y_test
        
    def train(self):
        """
        Scales data and trains the logistic regression model

        Returns:
            self.model
        """
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.model.fit(self.X_train, self.y_train)
        return self.model
        
    
    def predict(self):
        
        if self.X_test is None:
            print("Error: Test features must be prepared first")
        self.X_test = self.scaler.transform(self.X_test)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        return self.y_pred, self.y_pred_proba
    
    def evaluate(self):
        if self.y_test is None or self.y_pred is None:
            print("Error: Predictions must be made before evaluation")
            return None

        metrics_dict = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred),
            "f1_score": f1_score(self.y_test, self.y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred)
        }

        return metrics_dict
    
    
    def run_lr_pipeline(self):
        self.load_data()
        self.prepare_features()
        self.train()
        self.predict()
        return self.evaluate()