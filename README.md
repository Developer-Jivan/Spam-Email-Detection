# Intelligent Spam Email Detection using NLP & Machine Learning

This project builds an advanced spam email classifier using Natural Language Processing and machine learning models. It detects whether an email message is spam or legitimate using TF-IDF vectorization and optimized classifiers.

## Features

- Text preprocessing with lemmatization and stopword removal
- TF-IDF with bigrams
- Model comparison: Naive Bayes vs Linear SVM
- Hyperparameter tuning with GridSearch
- Evaluation with F1 score and ROC AUC
- Saved production model
- Interactive prediction app

## Tech Stack

Python, Scikit-learn, NLP, TF-IDF, SVM, Naive Bayes

## How it Works

1. Clean email text
2. Convert to TF-IDF vectors
3. Train multiple classifiers
4. Select best model
5. Save and deploy predictor

## Results

Achieved high precision and recall on spam detection with optimized vectorization and model tuning.

## Run

pip install -r requirements.txt
python src/train.py
python app.py
