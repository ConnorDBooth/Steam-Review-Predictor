#IMPORTS 
from preprocessing_data.pre_processing import PreProcessor
from feature_engineering.build_features import LogisticRegressionFeatureEngineering
from models.train_model import BasicLogisticRegressionTraining
from visualization.visualize import LogisticRegressionVisualizer

def main():
    processor = PreProcessor("src/data/raw/all_reviews.csv")

    df = processor.preprocess(5000000)
    
    feature_builder = LogisticRegressionFeatureEngineering(df)
    train_df, test_df = feature_builder.split_data()
    feature_builder.save_data(train_df, test_df)
    
    trainer = BasicLogisticRegressionTraining(train_df, test_df)
    results = trainer.run_lr_pipeline()
    
    print("\nModel Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
    
    plot = LogisticRegressionVisualizer(trainer.model, feature_builder.feature_cols )
    plot.plot_feature_importance()
    
if __name__ == "__main__":
    main()