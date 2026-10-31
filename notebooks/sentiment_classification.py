
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "cleaned_reviews.csv"

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("REVIEWRADAR AI - SENTIMENT CLASSIFICATION")
print("=" * 55)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading customer reviews...")

df = pd.read_csv(INPUT_FILE)

print("Total reviews:", len(df))

# --------------------------------------------------
# 3. PREPARE SENTIMENT LABELS
# --------------------------------------------------

print("\nPreparing sentiment labels...")

df = df[["Review_Text_Clean", "Rating"]].copy()

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df = df.dropna(subset=["Review_Text_Clean", "Rating"])

df["Review_Text_Clean"] = (
    df["Review_Text_Clean"].astype(str).str.strip()
)

df = df[df["Review_Text_Clean"] != ""]

# Convert ratings into sentiment categories
def assign_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["Sentiment"] = df["Rating"].apply(assign_sentiment)

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

# --------------------------------------------------
# 4. PREPARE FEATURES AND TARGET
# --------------------------------------------------

X = df["Review_Text_Clean"]
y = df["Sentiment"]

print("\nTotal usable reviews:", len(df))

# --------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

print("\nSplitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# --------------------------------------------------
# 6. BUILD MACHINE LEARNING PIPELINE
# --------------------------------------------------

print("\nBuilding TF-IDF + Logistic Regression model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\nTraining sentiment classification model...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 8. MAKE PREDICTIONS
# --------------------------------------------------

print("\nPredicting sentiment on test data...")

y_pred = model.predict(X_test)

# --------------------------------------------------
# 9. EVALUATE MODEL
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    labels=["Negative", "Neutral", "Positive"],
    output_dict=True,
    zero_division=0
)

print("\n" + "=" * 55)
print("MODEL EVALUATION RESULTS")
print("=" * 55)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"],
        zero_division=0
    )
)

# Save evaluation results
report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    REPORT_DIR / "sentiment_classification_report.csv"
)

# Save accuracy
metrics = {
    "model": "TF-IDF + Logistic Regression",
    "accuracy": round(float(accuracy), 4),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test))
}

with open(REPORT_DIR / "sentiment_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

print("\nGenerating confusion matrix...")

labels = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("ReviewRadar AI - Sentiment Confusion Matrix")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "sentiment_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 11. SAVE TRAINED MODEL
# --------------------------------------------------

MODEL_FILE = MODEL_DIR / "sentiment_model.pkl"

joblib.dump(model, MODEL_FILE)

print("\nTrained model saved successfully!")

# --------------------------------------------------
# 12. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 55)
print("SENTIMENT CLASSIFICATION COMPLETED!")
print("=" * 55)

print("Model:", MODEL_FILE)
print("Classification report:", REPORT_DIR / "sentiment_classification_report.csv")
print("Metrics:", REPORT_DIR / "sentiment_metrics.json")
print("Confusion matrix:", FIGURES_DIR / "sentiment_confusion_matrix.png")
