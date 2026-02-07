import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from preprocess import clean_text

# Load data
df = pd.read_csv("data/spam_data.csv")

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Clean text
df["clean"] = df["message"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

# Pipeline with TF-IDF + model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9
    )),
    ("clf", MultinomialNB())
])

params = {
    "clf": [MultinomialNB(), LinearSVC()],
}

grid = GridSearchCV(
    pipeline,
    param_grid=params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Model:", grid.best_params_)

preds = best_model.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(best_model, "models/spam_model.pkl")
print("Model saved.")
