#IMPORTS
import pandas as pd
import os
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    
    def votes_ratio(self):
        """
        Creates a ratio based on the number of helpful votes a review has,
        and the number of funny votes a review has
        """
        # +1 ensures we are not dividing by 0
        self.df["votes_ratio"] = self.df["votes_up"] / (self.df["votes_up"] + self.df["votes_funny"] + 1)
        self.feature_cols += ["votes_ratio"]
        
    def playtime_ratio(self):
        """
        Creates a playtime ratio feature from "author_playtime_at_review"
        and "author_playtime_forever" columns
        """
        self.df["playtime_ratio"] = self.df["author_playtime_at_review"] / (self.df["author_playtime_forever"])
        self.feature_cols += ["playtime_ratio"]
        
    def review_length(self):
        """
        Adds review length as a numeric feature
        """
        self.df["review_length"] = self.df["review"].str.split().str.len()
        self.feature_cols += ["review_length"]
    
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

from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFFeatureEngineering(LogisticRegressionFeatureEngineering):
    def __init__(self, df=None, max_features=100):
        super().__init__(df)
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.review_col = "review"

    def build_tfidf_features(self):
        """
        Generates TF-IDF features from the review column and appends
        them to the dataframe and feature_cols list.

        Returns:
            self.df: DataFrame with TF-IDF features added
        """
        if self.df is None:
            print("Error: No dataframe provided")
            return None

        if self.review_col not in self.df.columns:
            print(f"Error: '{self.review_col}' column not found")
            return None

        tfidf_matrix = self.vectorizer.fit_transform(self.df[self.review_col].fillna(""))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{w}" for w in self.vectorizer.get_feature_names_out()],
            index=self.df.index
        )
        self.df = pd.concat([self.df, tfidf_df], axis=1)
        self.feature_cols = self.feature_cols + list(tfidf_df.columns)

        return self.df

class LSTMFeatureEngineering:
    def __init__(self, df=None, max_words=2000, max_len=100):
        """
        Constructor for the "LSTMFeatureEngineering" class.
        Sets target variable and text column.
        oov_token = "<OOV>" in the Tokenizer function is used to 
        handle words in text data that where not part of training.

        Args:
            df: Input dataframe, Defaults to None.
            max_words:Limits how many unique words the model knows, Defaults to 2000.
            max_len: Controls the max length of each input sequence,  Defaults to 100.
        """
        self.df = df
        self.target_col = "voted_up"
        self.text_col = "review"
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        
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
        
        train_df, test_df = train_test_split(
            self.df[[self.text_col, self.target_col]],
            test_size=test_size,
            random_state=random_state,
            
            # Use stratify to preserve the class distribution of the target variable (voted_up)
            # in both the training and test sets. Without this, random splitting could create
            # imbalanced subsets (Ex. too many positive reviews), leading to unreliable
            # evaluation metrics and unfair comparisons between models.
            stratify=self.df[self.target_col]
        )
        
        return train_df, test_df
    
    def fit_transform(self, train_df, test_df):
        """
        Fits the tokenizer to the training data.
        Converts reviews to padded integer sequences.

        Args:
            train_df: training dataframe
            test_df: testing dataframe
            
        Return:
            X_train, X_test: Padded arrays ready for LSTM input
            y_train, y_test: Target arrays
        """
        #Fit tokenizer on training text
        self.tokenizer.fit_on_texts(train_df[self.text_col])
        
        X_train_seq = self.tokenizer.texts_to_sequences(train_df[self.text_col])
        X_test_seq = self.tokenizer.texts_to_sequences(test_df[self.text_col])
        
        #All inputs must be the same length, need to pad shorter inputs with 0s
        #'post' adds the additional 0's to the back (Ex. [15, 145, 0, 0, 0])
        X_train = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post', truncating='post')
        X_test = pad_sequences(X_test_seq, maxlen=self.max_len, padding="post", truncating="post")
        
        y_train = np.array(train_df[self.target_col])
        y_test= np.array(test_df[self.target_col])
        
        
        #Save tokenizer
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "../models")
        joblib.dump(self.tokenizer, os.path.join(models_dir, "lstm_tokenizer.pkl"))
        
        return X_train, X_test, y_train, y_test 