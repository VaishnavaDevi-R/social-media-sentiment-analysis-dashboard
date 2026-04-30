import joblib
import os
from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models/sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/vectorizer.pkl"))

def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return pred, prob

if __name__ == "__main__":
    text = input("Enter text: ")
    sentiment, confidence = predict(text)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")