
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "cleaned_reviews.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading customer reviews...")

df = pd.read_csv(INPUT_FILE)

print("Total reviews:", len(df))


# ==========================================
# 3. PREPARE REVIEW TEXT
# ==========================================

print("\nPreparing review text...")

# Remove rows with missing or empty cleaned reviews.
df["Review_Text_Clean"] = (
    df["Review_Text_Clean"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Keep only reviews containing actual text.
valid_reviews = df[
    df["Review_Text_Clean"].str.len() > 0
].copy()

print("Usable reviews:", len(valid_reviews))
print("Excluded empty reviews:", len(df) - len(valid_reviews))


# ==========================================
# 4. TF-IDF TEXT VECTORIZATION
# ==========================================

print("\nConverting reviews into numerical features...")

vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90
)

tfidf_matrix = vectorizer.fit_transform(
    valid_reviews["Review_Text_Clean"]
)

print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ==========================================
# 5. NMF TOPIC MODELLING
# ==========================================

print("\nDiscovering customer review topics...")

NUMBER_OF_TOPICS = 8

nmf_model = NMF(
    n_components=NUMBER_OF_TOPICS,
    random_state=42,
    init="nndsvda",
    max_iter=400
)

topic_matrix = nmf_model.fit_transform(tfidf_matrix)

print("Topic modelling completed!")


# ==========================================
# 6. EXTRACT TOPIC KEYWORDS
# ==========================================

print("\nExtracting topic keywords...")

feature_names = vectorizer.get_feature_names_out()

topic_rows = []

for topic_number, topic in enumerate(nmf_model.components_):

    top_indices = topic.argsort()[-10:][::-1]

    keywords = [
        feature_names[index]
        for index in top_indices
    ]

    topic_rows.append({
        "Topic_ID": topic_number + 1,
        "Top_Keywords": ", ".join(keywords)
    })

topic_summary = pd.DataFrame(topic_rows)

# ==========================================
# 6.1. BUSINESS-FRIENDLY TOPIC LABELS
# ==========================================

topic_labels = {
    1: "General Product Feedback",
    2: "Dress Design & Appearance",
    3: "Comfort & Wearability",
    4: "Size Accuracy & Fit",
    5: "Sizing & Size Selection",
    6: "Shirt Design & Material",
    7: "Sweater & Outerwear Feedback",
    8: "Product Quality & Appearance"
}

topic_summary["Topic_Label"] = (
    topic_summary["Topic_ID"].map(topic_labels)
)


# ==========================================
# 7. ASSIGN TOPICS TO REVIEWS
# ==========================================

print("\nAssigning topics to reviews...")

# Each review is assigned to the topic
# with the highest NMF topic weight.

valid_reviews["Topic_ID"] = (
    topic_matrix.argmax(axis=1) + 1
)

valid_reviews["Topic_Confidence_Score"] = (
    topic_matrix.max(axis=1)
)

# Save review-level topic assignments.
review_columns = [
    "Review_Text",
    "Review_Text_Clean",
    "Rating",
    "Recommended_IND",
    "Department_Name",
    "Class_Name",
    "Topic_ID",
    "Topic_Confidence_Score"
]

# Retain only columns that exist in the dataset.
review_columns = [
    column for column in review_columns
    if column in valid_reviews.columns
]

review_results = valid_reviews[review_columns].copy()


# ==========================================
# 8. CALCULATE TOPIC DISTRIBUTION
# ==========================================

print("\nCalculating topic distribution...")

topic_counts = (
    review_results["Topic_ID"]
    .value_counts()
    .sort_index()
)

topic_distribution = pd.DataFrame({
    "Topic_ID": topic_counts.index,
    "Review_Count": topic_counts.values
})

topic_distribution["Percentage"] = (
    topic_distribution["Review_Count"]
    / len(review_results)
    * 100
).round(2)


# ==========================================
# 8.1. TOPIC SATISFACTION ANALYSIS
# ==========================================

topic_satisfaction = (
    review_results
    .groupby("Topic_ID")
    .agg(
        Review_Count=("Rating", "count"),
        Average_Rating=("Rating", "mean"),
        Non_Recommended_Count=(
            "Recommended_IND",
            lambda x: (x == 0).sum()
        )
    )
    .reset_index()
)

topic_satisfaction["Non_Recommended_Percentage"] = (
    topic_satisfaction["Non_Recommended_Count"]
    / topic_satisfaction["Review_Count"]
    * 100
).round(2)

topic_satisfaction["Average_Rating"] = (
    topic_satisfaction["Average_Rating"].round(2)
)

topic_satisfaction = topic_satisfaction.merge(
    topic_summary[["Topic_ID", "Topic_Label"]],
    on="Topic_ID",
    how="left"
)

topic_satisfaction.to_csv(
    OUTPUT_DIR / "topic_satisfaction.csv",
    index=False
)

print("\nTOPIC SATISFACTION ANALYSIS:")

print(topic_summary.to_string(index=False, max_colwidth=35))

topic_satisfaction.to_csv(
    OUTPUT_DIR / "topic_satisfaction.csv",
    index=False
)

print("Topic satisfaction saved successfully!")

# ==========================================
# 9. FIND REPRESENTATIVE REVIEWS
# ==========================================

print("\nFinding representative reviews...")

# Select the highest-weight review for each topic.
representative_reviews = (
    review_results
    .sort_values(
        "Topic_Confidence_Score",
        ascending=False
    )
    .groupby("Topic_ID")
    .head(5)
    .sort_values(
        ["Topic_ID", "Topic_Confidence_Score"],
        ascending=[True, False]
    )
)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

topic_summary = topic_summary.merge(
    topic_distribution,
    on="Topic_ID",
    how="left"
)

topic_summary.to_csv(
    OUTPUT_DIR / "topic_summary.csv",
    index=False
)

review_results.to_csv(
    OUTPUT_DIR / "review_topic_assignments.csv",
    index=False
)

representative_reviews.to_csv(
    OUTPUT_DIR / "representative_reviews.csv",
    index=False
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 55)
print("REVIEWRADAR AI - TOPIC ANALYSIS RESULTS")
print("=" * 55)

print("\nDISCOVERED TOPICS:")

print(topic_summary.to_string(index=False, max_colwidth=35))

print("\nREPRESENTATIVE REVIEWS:")

for topic_id in sorted(
    representative_reviews["Topic_ID"].unique()
):

    print(f"\nTOPIC {topic_id}")

    examples = representative_reviews[
        representative_reviews["Topic_ID"] == topic_id
    ]

    for _, row in examples.head(3).iterrows():

        print("-", str(row["Review_Text_Clean"])[:250])


print("\nAnalysis completed successfully!")

print("\nGenerated files:")
print("- topic_summary.csv")
print("- review_topic_assignments.csv")
print("- representative_reviews.csv")
