
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_reviews.csv")

# 2. Keep reviews with usable text
df = df.dropna(subset=["Review_Text_Clean"]).copy()
df = df[df["Review_Text_Clean"].str.strip() != ""]

# 3. Create rating-based sentiment labels
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

# 4. Recreate the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Load the complete selected model pipeline
# The pipeline already contains its TF-IDF vectorizer.
model = joblib.load("models/best_sentiment_model_day5.pkl")

# 6. Predict using raw review text
y_pred = model.predict(X_test)

# 7. Create error-analysis dataset
results = pd.DataFrame({
    "Review": X_test.values,
    "Actual_Sentiment": y_test.values,
    "Predicted_Sentiment": y_pred
})

errors = results[
    results["Actual_Sentiment"] != results["Predicted_Sentiment"]
].copy()

errors.to_csv("reports/misclassified_reviews.csv", index=False)

# 8. Display evaluation results
labels = ["Negative", "Neutral", "Positive"]

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred, labels=labels))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, y_pred, labels=labels, zero_division=0
))

print("\nERROR ANALYSIS")
print("Total test reviews:", len(results))
print("Incorrect predictions:", len(errors))
print("Correct predictions:", len(results) - len(errors))
print("Error rate:", round(len(errors) / len(results) * 100, 2), "%")

print("\nMISCLASSIFICATION BREAKDOWN")
if errors.empty:
    print("No misclassified reviews were found.")
else:
    print(pd.crosstab(
        errors["Actual_Sentiment"],
        errors["Predicted_Sentiment"]
    ))

print("\nEXAMPLE MISCLASSIFIED REVIEWS")
if errors.empty:
    print("No examples to display.")
else:
    for _, row in errors.head(10).iterrows():
        print("\nReview:", row["Review"])
        print("Actual:", row["Actual_Sentiment"])
        print("Predicted:", row["Predicted_Sentiment"])

print("\nError analysis completed!")
