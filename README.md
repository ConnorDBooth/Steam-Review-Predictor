# Steam Review Sentiment Classifier

A machine learning pipeline that classifies Steam game reviews as **positive** or **negative** — built as a course project for CP-322 (Machine Learning) at Wilfrid Laurier University.

---

## What It Does

Given a Steam review like *"10/10 would die again"*, the model predicts whether it's a positive or negative review. We trained and compared four models across two feature representations to see what actually works best on messy, informal game review text.

---

## Models & Results

| Model | Features | F1 Score |
|---|---|---|
| Logistic Regression | Numeric | **0.7836** |
| Logistic Regression | TF-IDF | **0.8756** |
| Random Forest | TF-IDF | **0.8599** |
| LSTM | Sequential text | **0.9250** |

> The LSTM edged out TF-IDF + LR in F1, but the gap was surprisingly small for the added complexity.

## Stack

- **Python** — scikit-learn, pandas, Keras/TensorFlow
- **Models** — Logistic Regression, Random Forest, LSTM
- **Features** — TF-IDF vectors, numeric review features, sequential embeddings
- **Report** — Quarto (rendered to PDF)

---
## Authors

- **Connor Booth** — [github.com/ConnorDBooth](https://github.com/ConnorDBooth)
- **Filip Droca**

Built for CP-322 · Wilfrid Laurier University
Project Organization
------------

    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention should snake case.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       ├── data
       │   ├── processed      <- The final, canonical data sets for modeling.
       │   └── raw            <- The original, immutable data dump.
       │
       ├── preprocessing_data           <- Scripts to download or generate data and pre-process the data
       │   └── pre-processing.py
       │
       ├── feature_engineering       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
       │   └── visualize.py  
       │
       └── main.py  <- main script to run all the models and call appropriate functions
       |
       ├── LICENSE  <- LICENSE terms to be included for the use of the source code distribution



