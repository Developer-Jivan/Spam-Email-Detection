import joblib
from preprocess import clean_text

model = joblib.load("models/spam_model.pkl")

def predict_email(text):
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]

    return "SPAM" if pred == 1 else "HAM"


if __name__ == "__main__":
    msg = input("Enter email text: ")
    print(predict_email(msg))
