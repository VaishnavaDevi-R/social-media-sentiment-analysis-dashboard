import pandas as pd
import joblib
import os
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fix working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "data/raw/sample_raw.csv"))

# Clean text
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
model_path = os.path.join(BASE_DIR, "models/sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models/vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("🔥 Model trained and saved successfully!")