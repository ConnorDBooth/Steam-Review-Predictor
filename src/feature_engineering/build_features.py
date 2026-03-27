#IMPORTS
import pandas as pd
from sklearn.model_selection import train_test_split

#CONSTANTS
RANDOM_STATE = 42

class LogisticRegressionFeatureEngineering:
    def __init__(self, df=None):
        """
        Constructor for the "LogisticRegressionFeatureEngineering" class.
        Accepts the raw dataframe and configures X and Y variables. 

        Args:
            df (_type_, optional): _description_. Defaults to None.
        """
        self.df = df
        self.target_col = "voted_up"
        
        self.feature_cols = [
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
        self.train_path = None
        self.test_path = None
        
    
    def split_data(self, test_size=0.2, random_state=RANDOM_STATE):
        """
        Creates Train and Test splits of the data. 

        Args:
            test_size: Defaults to 0.2.
            random_state: Defaults to RANDOM_STATE.

        Returns:
            train_df: Dataframe of training data
            test_df: Dataframe of testing data
        """
        if self.df is None:
            print("Error: No dataframe provided")
            return None, None
        
        relevant_df = self.df[self.feature_cols + [self.target_col]]
        
        train_df, test_df = train_test_split(
            relevant_df,
            test_size=test_size,
            random_state=random_state,
            
            # Use stratify to preserve the class distribution of the target variable (voted_up)
            # in both the training and test sets. Without this, random splitting could create
            # imbalanced subsets (Ex. too many positive reviews), leading to unreliable
            # evaluation metrics and unfair comparisons between models.
            stratify=self.df[self.target_col]
        )
        
        return train_df, test_df
    
    def save_data(self, train_df, test_df, train_path="src/data/processed/train.csv",test_path="src/data/processed/test.csv"):
        """
        Saves training and test data to src/data/processed"
        """
        if train_df is None or test_df is None:
            print("Error: Train/test data is not present")
            return
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        self.train_path = train_path
        self.test_path = test_path
        
        print(f"Training data saved to: {train_path}")
        print(f"Testing Data saved to: {test_path}")
    
    def load_data(self):
        """
        Loads and sets the training and testing dataframes
        
        Returns:
            Training and testing dataframes
        """
        try:
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            return train_df, test_df
        except FileNotFoundError as e :
            print(f"Error: could not load data: {e}")
            return None, None
        