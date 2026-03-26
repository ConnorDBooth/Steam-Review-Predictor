#IMPORTS
import pandas as pd
from sklearn.model_selection import train_test_split

#CONSTANTS
RANDOM_STATE = 42
class PreProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
    def load_data(self, nrows=None):
        """
        Converts the raw data into a Pandas DataFrame 
        
        Args: nrows
            -Specifies the number of rows from the dataset to be used 
            in the dataframe. 
        Returns:
            self.df: Dataframe

        """
        try:
            self.df = pd.read_csv(self.filepath, nrows=nrows)
            return self.df
        except FileNotFoundError:
            print(f"Error: The file at {self.filepath} could not be found")
            return None
        
    def remove_uninteresting_columns(self):
        """
        Removes any columns that will not be useful during training.
        Feel free to update as needed. 
        
        Returns:
            self.df: Dataframe

        """
        self.df = self.df.drop(columns= [
            "recommendationid",
            "appid",
            "hidden_in_steam_china",
            "steam_china_location",
        ])
        
        return self.df
    
    def remove_no_playtime(self):
        """
        Removes any entries where the author has not played the game
        
        Returns:
            self.df: Dataframe

        """
        self.df = self.df[self.df["author_playtime_forever"] > 0]
        
        return self.df

    def remove_empty_reviews(self):
        """
        Removes any entries where the review section is empty
        
        Returns:
            self.df: Dataframe

        """
        if "review" in self.df.columns:
            self.df = self.df.dropna(subset=["review"])
            self.df = self.df[self.df["review"].str.strip() != ""]
        return self.df
    
    def handle_missing_numeric_values(self):
        """
        Fills any NaN numeric column with 0, indicating False or none.
        Does not include the target variable or timestamp columns
        
        Returns:
            self.df: Dataframe

        """
        #List of numeric columns present in the dataset 
        numeric_cols = [
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
            "written_during_early_access"
        ]
        
        #Find columns within the above list that actually appear in the dataset
        #Will prevent errors if any of these cols are removed in the future
        valid_cols = self.df.columns.intersection(numeric_cols)
        
        #Fill NaN values with 0 for each col 
        self.df[valid_cols] = self.df[valid_cols].fillna(0)
        
        return self.df
        
    def handle_missing_target_variable(self):
        """
        Handles missing target variable.
        If target variable is missing, the row is removed

        Returns:
            self.df: Dataframe
        """
        if "voted_up" in self.df.columns:
            self.df = self.df.dropna(subset=["voted_up"])
        return self.df
    
    def handle_timestamps(self):
        """
        Converts Unix timestamp columns to datetime.
        Missing values are left as NaT rather than filled with 0.

        Returns:
            self.df: DataFrame
        """
        timestamp_columns = [
            "timestamp_created",
            "timestamp_updated",
            "author_last_played"
        ]
        
        valid_cols = self.df.columns.intersection(timestamp_columns)
        
        self.df[valid_cols] = self.df[valid_cols].apply(pd.to_datetime, unit="s", errors="coerce")
        
        return self.df
    
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
        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            
            # Use stratify to preserve the class distribution of the target variable (voted_up)
            # in both the training and test sets. Without this, random splitting could create
            # imbalanced subsets (Ex. too many positive reviews), leading to unreliable
            # evaluation metrics and unfair comparisons between models.
            stratify=self.df["voted_up"]
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
        
        print(f"Training data saved to: {train_path}")
        print(f"Testing Data saved to: {test_path}")
    """
    Function to remove all non-English reviews. 
    Can be removed if we end up using xlmr, 
    otherwise uncomment
        
    def filter_english_reviews(self):
        if "language" in self.df.columns:
            self.df = self.df[self.df["language"] == "english"]
            
        return self.df
        
    """
    
    def preprocess(self, nrows=None):
        """
        Runs the full preprocessing pipeline.

        Args:
            nrows: Number of rows from the dataset to load.

        Returns:
            self.df: DataFrame
        """
        self.load_data(nrows=nrows)
        self.remove_uninteresting_columns()
        self.remove_no_playtime()
        self.remove_empty_reviews()
        self.handle_missing_numeric_values()
        self.handle_missing_target_variable()
        self.handle_timestamps()
        
        return self.df
        
        