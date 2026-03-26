#IMPORTS 
from preprocessing_data.pre_processing import PreProcessor



def main():
    processor = PreProcessor("src/data/raw/all_reviews.csv")

    processor.preprocess(100000)
    
    train_df, test_df = processor.split_data()
    
    processor.save_data(train_df, test_df)
    
    
if __name__ == "__main__":
    main()