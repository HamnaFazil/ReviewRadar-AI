
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_DIR / "data" / "raw" / "clothing_reviews.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("REVIEWRADAR AI - DATA CLEANING AND EDA")
print("=" * 60)

print(f"\nLooking for dataset at:\n{RAW_FILE}")

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found: {RAW_FILE}\n"
        "Place clothing_reviews.csv inside data/raw/."
    )

df = pd.read_csv(RAW_FILE)

print("\nDataset loaded successfully!")
print(f"Original rows: {df.shape[0]:,}")
print(f"Original columns: {df.shape[1]}")


# ============================================================
# 3. INSPECT DATASET
# ============================================================

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values before cleaning:")
print(df.isnull().sum().sort_values(ascending=False))

print(f"\nDuplicate rows before cleaning: {df.duplicated().sum()}")


# ============================================================
# 4. STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

print("\nStandardized column names:")
print(df.columns.tolist())


# ============================================================
# 5. REMOVE DUPLICATE ROWS
# ============================================================

rows_before = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = rows_before - len(df)

print(f"\nDuplicate rows removed: {duplicates_removed}")


# ============================================================
# 6. CLEAN REVIEW TEXT
# ============================================================

if "Review_Text" in df.columns:

    # Preserve the original review text.
    df["Review_Text"] = (
        df["Review_Text"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # Create a normalized version for NLP.
    df["Review_Text_Clean"] = (
        df["Review_Text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    missing_reviews = df["Review_Text"].isna().sum()

    print(f"\nMissing review texts: {missing_reviews:,}")

    print(
        "Rows with missing review text retained:",
        missing_reviews,
    )

else:
    print(
        "\nWARNING: Review_Text column not found. "
        "Check the dataset column names."
    )


# ============================================================
# 7. CHECK RATINGS AND RECOMMENDATIONS
# ============================================================

if "Rating" in df.columns:

    df["Rating"] = pd.to_numeric(
        df["Rating"], errors="coerce"
    )

    print("\nRating distribution:")
    print(df["Rating"].value_counts(dropna=False).sort_index())

    print("\nAverage rating:")
    print(round(df["Rating"].mean(), 2))

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Rating",
        order=sorted(df["Rating"].dropna().unique()),
    )

    plt.title("Customer Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rating_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

else:
    print("\nWARNING: Rating column not found.")


if "Recommended_IND" in df.columns:

    print("\nRecommendation distribution:")
    print(
        df["Recommended_IND"]
        .value_counts(dropna=False)
        .sort_index()
    )

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Recommended_IND",
        order=sorted(
            df["Recommended_IND"].dropna().unique()
        ),
    )

    plt.title("Customer Recommendation Distribution")
    plt.xlabel("Recommendation (0 = No, 1 = Yes)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "recommendation_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

else:
    print("\nWARNING: Recommended_IND column not found.")


# ============================================================
# 8. ANALYZE PRODUCT DEPARTMENTS
# ============================================================

if {"Department_Name", "Rating"}.issubset(df.columns):

    department_summary = (
        df.groupby("Department_Name", dropna=False)
        .agg(
            Review_Count=("Rating", "count"),
            Average_Rating=("Rating", "mean"),
        )
        .sort_values("Average_Rating", ascending=False)
    )

    print("\nDepartment analysis:")
    print(department_summary.round(2))

    department_summary.to_csv(
        PROCESSED_DIR / "department_summary.csv"
    )

    plot_data = department_summary.dropna(
        subset=["Average_Rating"]
    )

    if not plot_data.empty:

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=plot_data.reset_index(),
            x="Department_Name",
            y="Average_Rating",
        )

        plt.title("Average Customer Rating by Department")
        plt.xlabel("Department")
        plt.ylabel("Average Rating")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "department_ratings.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# 9. ANALYZE RATINGS BY PRODUCT CLASS
# ============================================================

if {"Class_Name", "Rating"}.issubset(df.columns):

    class_summary = (
        df.groupby("Class_Name", dropna=False)
        .agg(
            Review_Count=("Rating", "count"),
            Average_Rating=("Rating", "mean"),
        )
        .sort_values("Average_Rating")
    )

    print("\nProduct class analysis:")
    print(class_summary.round(2))

    class_summary.to_csv(
        PROCESSED_DIR / "class_summary.csv"
    )

    plot_data = class_summary.dropna(
        subset=["Average_Rating"]
    )

    if not plot_data.empty:

        plt.figure(figsize=(12, 7))

        sns.barplot(
            data=plot_data.reset_index(),
            x="Average_Rating",
            y="Class_Name",
        )

        plt.title("Average Customer Rating by Product Class")
        plt.xlabel("Average Rating")
        plt.ylabel("Product Class")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "class_ratings.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# 10. ANALYZE POSITIVE FEEDBACK COUNTS
# ============================================================

if "Positive_Feedback_Count" in df.columns:

    df["Positive_Feedback_Count"] = pd.to_numeric(
        df["Positive_Feedback_Count"],
        errors="coerce",
    )

    print("\nPositive feedback count summary:")
    print(
        df["Positive_Feedback_Count"]
        .describe()
        .round(2)
    )


# ============================================================
# 11. SAVE CLEANED DATASET
# ============================================================

output_file = PROCESSED_DIR / "cleaned_reviews.csv"

df.to_csv(output_file, index=False)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CLEANING AND EDA COMPLETED")
print("=" * 60)

print(f"Original dataset: {RAW_FILE}")
print(f"Rows after cleaning: {len(df):,}")
print(f"Columns after cleaning: {len(df.columns)}")
print(f"Duplicate rows removed: {duplicates_removed}")

if "Review_Text" in df.columns:
    print(
        "Rows with missing review text:",
        df["Review_Text"].isna().sum(),
    )

print(f"\nCleaned dataset saved to:\n{output_file}")

print(f"\nCharts saved to:\n{FIGURES_DIR}")

print("\nGenerated files:")

for file_path in sorted(PROCESSED_DIR.glob("*.csv")):
    print(f"  - {file_path.name}")

for file_path in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  - {file_path.name}")

print("\nNext milestone: identify recurring customer complaints.")
