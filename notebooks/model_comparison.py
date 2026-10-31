
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "processed" / "cleaned_reviews.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("REVIEWRADAR AI - DAY 5 MODEL COMPARISON")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading customer reviews...")

df = pd.read_csv(DATA_FILE)

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df = df.dropna(subset=["Review_Text_Clean", "Rating"])

df["Review_Text_Clean"] = df["Review_Text_Clean"].astype(str).str.strip()

df = df[df["Review_Text_Clean"] != ""]

# --------------------------------------------------
# 3. CREATE SENTIMENT LABELS
# --------------------------------------------------

def assign_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


df["Sentiment"] = df["Rating"].apply(assign_sentiment)

X = df["Review_Text_Clean"]
y = df["Sentiment"]

print("\nSentiment distribution:")
print(y.value_counts())

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Linear SVC": LinearSVC(
        class_weight="balanced",
        random_state=42
    ),

    "Complement Naive Bayes": ComplementNB(
        alpha=0.5
    )
}

results = []
classification_reports = {}
trained_models = {}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    pipeline = Pipeline([
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
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })

    classification_reports[model_name] = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    trained_models[model_name] = pipeline

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# --------------------------------------------------
# 7. COMPARE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

# --------------------------------------------------
# 8. SAVE CLASSIFICATION REPORTS
# --------------------------------------------------

report_rows = []

for model_name, report in classification_reports.items():

    for sentiment in ["Negative", "Neutral", "Positive"]:

        report_rows.append({
            "Model": model_name,
            "Sentiment": sentiment,
            "Precision": report[sentiment]["precision"],
            "Recall": report[sentiment]["recall"],
            "F1_Score": report[sentiment]["f1-score"],
            "Support": report[sentiment]["support"]
        })

reports_df = pd.DataFrame(report_rows)

reports_df.to_csv(
    REPORT_DIR / "model_classification_details.csv",
    index=False
)

# --------------------------------------------------
# 9. SELECT BEST MODEL USING MACRO F1
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    MODEL_DIR / "best_sentiment_model_day5.pkl"
)

print("\nBest model:", best_model_name)

# --------------------------------------------------
# 10. SAVE BEST MODEL METRICS
# --------------------------------------------------

best_metrics = results_df.iloc[0].to_dict()

best_metrics["Best_Model"] = best_model_name

with open(
    REPORT_DIR / "best_model_metrics.json",
    "w"
) as file:
    json.dump(best_metrics, file, indent=4)

# --------------------------------------------------
# 11. GENERATE MODEL COMPARISON CHART
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Macro_F1"]
)

plt.title("Model Comparison - Macro F1 Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("Macro F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "model_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 12. GENERATE BEST MODEL CONFUSION MATRIX
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    labels=["Negative", "Neutral", "Positive"],
    cmap="Blues",
    xticks_rotation="vertical"
)

plt.title(f"Best Model Confusion Matrix - {best_model_name}")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "best_model_confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 13. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("DAY 5 MODEL COMPARISON COMPLETED!")
print("=" * 60)

print("\nBest model:", best_model_name)

print("\nGenerated reports:")
print("reports/model_comparison.csv")
print("reports/model_classification_details.csv")
print("reports/best_model_metrics.json")
print("reports/figures/model_comparison.png")
print("reports/figures/best_model_confusion_matrix.png")
print("models/best_sentiment_model_day5.pkl")
