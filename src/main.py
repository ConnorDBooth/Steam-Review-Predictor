#IMPORTS 
from preprocessing_data.pre_processing import PreProcessor
from feature_engineering.build_features import LogisticRegressionFeatureEngineering, TFIDFFeatureEngineering
from models.train_model import BasicLogisticRegressionTraining, BasicRandomForestTraining
from visualization.visualize import LogisticRegressionVisualizer, RandomForestVisualizer, TFIDFVisualizer

def main():
    processor = PreProcessor("src/data/raw/all_reviews.csv")
    df = processor.preprocess(1000000)
    
    feature_builder = LogisticRegressionFeatureEngineering(df)
    base_train_df, base_test_df = feature_builder.split_data()
    feature_builder.save_data(base_train_df, base_test_df)
    
    lr_trainer = BasicLogisticRegressionTraining(base_train_df, base_test_df)
    lr_results = lr_trainer.run_lr_pipeline()
    
    print("\nLogistic Regression Results:")
    for k, v in lr_results.items():
        print(f"{k}: {v}")
    
    plot = LogisticRegressionVisualizer(lr_trainer.model, feature_builder.feature_cols)
    plot.plot_feature_importance()

    tfidf_builder = TFIDFFeatureEngineering(df, max_features=1000)
    tfidf_builder.build_tfidf_features()
    tfidf_train_df, tfidf_test_df = tfidf_builder.split_data()
    tfidf_builder.save_data(tfidf_train_df, tfidf_test_df,
        train_path="src/data/processed/tfidf_train.csv",
        test_path="src/data/processed/tfidf_test.csv"
    )
    tfidf_trainer = BasicLogisticRegressionTraining(tfidf_train_df, tfidf_test_df, feature_engineer=tfidf_builder)
    tfidf_results = tfidf_trainer.run_lr_pipeline()

    print("\nTF-IDF Logistic Regression Results:")
    for k, v in tfidf_results.items():
        print(f"{k}: {v}")

    tfidf_plot = TFIDFVisualizer(tfidf_trainer.model, tfidf_builder.feature_cols)
    tfidf_plot.plot_feature_importance(top_n=50)

    # Random Forest
    rf_trainer = BasicRandomForestTraining(base_train_df, base_test_df)
    rf_results = rf_trainer.run_rf_pipeline()

    print("\nRandom Forest Results:")
    for k, v in rf_results.items():
        print(f"{k}: {v}")
    

    rf_plot = RandomForestVisualizer(rf_trainer.model, feature_builder.feature_cols)
    rf_plot.plot_feature_importance()

if __name__ == "__main__":
    main()