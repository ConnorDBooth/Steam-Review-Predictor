#IMPORTS 
from preprocessing_data.pre_processing import PreProcessor
from feature_engineering.build_features import LogisticRegressionFeatureEngineering, TFIDFFeatureEngineering,LSTMFeatureEngineering
from models.train_model import BasicLogisticRegressionTraining, BasicRandomForestTraining,LSTMTraining
from visualization.visualize import LogisticRegressionVisualizer, RandomForestVisualizer, TFIDFVisualizer,LSTMVisualizer, ModelComparisonVisualizer

def main():
    processor = PreProcessor("src/data/raw/balanced_10m_reviews.csv")
    df = processor.preprocess(100000)
    
    feature_builder = LogisticRegressionFeatureEngineering(df)
    #Save original feature list
    original_feature_cols = feature_builder.feature_cols.copy()
    base_train_df, base_test_df = feature_builder.split_data()
    #Add new features for LR only
    feature_builder.votes_ratio()
    feature_builder.playtime_ratio()
    feature_builder.review_length()
    lr_train_df, lr_test_df = feature_builder.split_data()
    feature_builder.save_data(base_train_df, base_test_df)
    
    lr_trainer = BasicLogisticRegressionTraining(lr_train_df, lr_test_df, feature_builder)
    lr_results = lr_trainer.run_lr_pipeline()
    
    print("\nLogistic Regression Results:")
    for k, v in lr_results.items():
        print(f"{k}: {v}")
    
    plot = LogisticRegressionVisualizer(lr_trainer.model, feature_builder.feature_cols)
    plot.plot_all(lr_trainer.y_test, lr_trainer.y_pred)
    
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
    

    rf_plot = RandomForestVisualizer(rf_trainer.model, original_feature_cols)
    rf_plot.plot_all(rf_trainer.y_test, rf_trainer.y_pred)

    #LSTM
    lstm_builder = LSTMFeatureEngineering(df)
    lstm_train_df, lstm_test_df = lstm_builder.split_data()
    lstm_trainer = LSTMTraining(lstm_train_df, lstm_test_df)
    lstm_results = lstm_trainer.run_lstm_pipeline()
    print("\nLSTM Results:")
    for k, v in lstm_results.items():
        print(f"{k}: {v}")
        
    lstm_plot = LSTMVisualizer(lstm_trainer.model, lstm_trainer.history, lstm_trainer.y_test, lstm_trainer.y_pred, lstm_trainer.y_pred_proba)
    lstm_plot.plot_all()
    
    
    comparison = ModelComparisonVisualizer({
        "Logistic Regression": lr_results,
        "Random Forest": rf_results,
        "TF-IDF LR": tfidf_results,
        "LSTM": lstm_results
    })
    comparison.plot_comparison()
    
if __name__ == "__main__":
    main()