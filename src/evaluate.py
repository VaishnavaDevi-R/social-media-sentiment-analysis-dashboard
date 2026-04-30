import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model & vectorizer
model = joblib.load(os.path.join(BASE_DIR, "models/sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/vectorizer.pkl"))

# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "data/raw/sample_raw.csv"))

X = df['text']
y = df['sentiment']

# Transform
X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

# ---------------- CONFUSION MATRIX ----------------
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(model, X_vec, y)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(BASE_DIR, "outputs/confusion_matrix.png"))
plt.close()

# ---------------- ACCURACY ----------------
accuracy = accuracy_score(y, y_pred)

# ---------------- CLASSIFICATION REPORT ----------------
report = classification_report(y, y_pred)

# Save report
with open(os.path.join(BASE_DIR, "outputs/model_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

# ---------------- SENTIMENT DISTRIBUTION ----------------
sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig(os.path.join(BASE_DIR, "outputs/sentiment_distribution.png"))
plt.close()

print("🔥 Evaluation completed! Files saved in outputs/")