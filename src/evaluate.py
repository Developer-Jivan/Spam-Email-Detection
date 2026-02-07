import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, roc_auc_score
from preprocess import clean_text

df = pd.read_csv("data/spam_data.csv")
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["clean"] = df["message"].apply(clean_text)

model = joblib.load("models/spam_model.pkl")

preds = model.predict(df["clean"])

print("Confusion Matrix")
print(confusion_matrix(df["label"], preds))

if hasattr(model, "decision_function"):
    scores = model.decision_function(df["clean"])
    print("ROC AUC:", roc_auc_score(df["label"], scores))
