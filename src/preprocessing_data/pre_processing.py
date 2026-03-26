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
        Handles missing target variable(s).
        If target variable(s) is/are missing, the row is removed.

        Returns:
            self.df: Dataframe
        """
        if "voted_up" in self.df.columns:
            self.df = self.df.dropna(subset=["voted_up"])
        if "game" in self.df.columns:
            self.df = self.df.dropna(subset=['game'])
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
    
    def filter_english_reviews(self):
        """
        Function to remove all non-English reviews. 
        Can be removed if we end up using xlmr, 
        otherwise uncomment
        """    
        if "language" in self.df.columns:
            self.df = self.df[self.df["language"] == "english"]
            
        return self.df
        
    def summarize(self):
        """
        Function to print statistics about current dataframe.
        """
        print('-------------------------')

        if self.df is None:
            print("No data loaded yet.")
            return

        rows, cols = self.df.shape
        print(f"Current Dataframe has {rows:,} rows and {cols} columns.")

        #missing vals
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("No missing values found.")
        else:
            print(f"\nMissing values in {len(missing)} column(s):")
            for col, count in missing.items():
                print(f"  {col}: {count:,} ({count / rows * 100:.1f}% missing)")

        #positive to negative ratio
        if "voted_up" in self.df.columns:
            counts = self.df["voted_up"].value_counts()
            pos = counts.get(True, counts.get(1, counts.get(1.0, 0)))
            neg = counts.get(False, counts.get(0, counts.get(0.0, 0)))
            print(f"\n{pos:,} reviews are positive ({pos / rows * 100:.1f}%) and {neg:,} are negative ({neg / rows * 100:.1f}%).")

        #playtime
        if "author_playtime_forever" in self.df.columns:
            pt = self.df["author_playtime_forever"]
            print(f"\nOn average, authors have played for {pt.mean():,.0f} minutes (median: {pt.median():,.0f}, longest: {pt.max():,.0f}).")

        if "author_playtime_at_review" in self.df.columns:
            pt_review = self.df["author_playtime_at_review"]
            print(f"At the time of writing, authors had played for {pt_review.mean():,.0f} minutes on average (median: {pt_review.median():,.0f}, longest: {pt_review.max():,.0f}).")

        #review stats
        if "review" in self.df.columns:
            lengths = self.df["review"].dropna().str.split().str.len()
            print(f"\nReviews are {lengths.mean():,.1f} words long on average (median: {lengths.median():,.0f}, longest: {lengths.max():,.0f} words).")

        #review author stats
        if "author_num_reviews" in self.df.columns:
            print(f"\nAuthors have written {self.df['author_num_reviews'].mean():,.1f} reviews on average (median: {self.df['author_num_reviews'].median():,.0f}, most prolific: {self.df['author_num_reviews'].max():,.0f}).")

        if "author_num_games_owned" in self.df.columns:
            print(f"Authors own {self.df['author_num_games_owned'].mean():,.1f} games on average (median: {self.df['author_num_games_owned'].median():,.0f}, most: {self.df['author_num_games_owned'].max():,.0f}).")

        #purchase flags
        flags = {
            "steam_purchase": "purchased on Steam",
            "received_for_free": "received for free",
            "written_during_early_access": "written during early access"
        }
        present_flags = {label: col for col, label in flags.items() if col in self.df.columns}
        if present_flags:
            print("\nOf the reviews in this dataframe:")
            for label, col in present_flags.items():
                count = int(self.df[col].sum())
                print(f"  {count:,} ({count / rows * 100:.1f}%) were {label}.")

        #games contained in set
        if "game" in self.df.columns:
            print(f"\nThe current dataframe covers {self.df['game'].nunique():,} unique game(s).")
        
        print('-------------------------')

    def remove_duplicate_reviews(self):
        """
        Function to remove repeats reviews of the same game
        from the same author. Will keep the first review.

        Returns:
            self.df: DataFrame
        """
        if "author_steamid" in self.df.columns and "timestamp_created" in self.df.columns and "game" in self.df.columns:
            self.df = self.df.sort_values("timestamp_created", ascending=False)
            self.df = self.df.drop_duplicates(subset=["author_steamid", "game"], keep="first")
        return self.df

    def remove_playtime_outliers(self, max_hours=30000):
        if "author_playtime_forever" in self.df.columns:
            self.df = self.df[self.df["author_playtime_forever"] <= max_hours * 60]
        return self.df
    
    def clean_review_text(self):
        """
        Function to standardize review text for future tokenization.

        Returns:
            self.df: DataFrame
        """
        if "review" in self.df.columns:
            #html
            self.df["review"] = self.df["review"].str.replace(r"https?://\S+", "", regex=True)
            self.df["review"] = self.df["review"].str.replace(r"\[.*?\]", "", regex=True)
            #special chars
            self.df["review"] = self.df["review"].str.replace(r"[^a-zA-Z0-9\s.,!?']", "", regex=True)
            #whitespace
            self.df["review"] = self.df["review"].str.strip().str.replace(r"\s+", " ", regex=True)
        return self.df
    
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
        self.filter_english_reviews()
        self.remove_no_playtime()
        self.remove_playtime_outliers()
        self.remove_duplicate_reviews()
        self.clean_review_text()
        self.remove_empty_reviews()
        self.handle_missing_numeric_values()
        self.handle_missing_target_variable()
        self.handle_timestamps()
        self.summarize()
        
        return self.df
        
        