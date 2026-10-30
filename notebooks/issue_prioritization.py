
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "topic_satisfaction.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

print("\nREVIEWRADAR AI - ISSUE PRIORITIZATION")
print("=" * 50)

# Load topic satisfaction data
df = pd.read_csv(INPUT_FILE)

# Check required columns
required_columns = [
    "Topic_ID",
    "Topic_Label",
    "Review_Count",
    "Average_Rating",
    "Non_Recommended_Percentage"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Normalize values to a 0-100 scale
def normalize(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(0, index=series.index)

    return ((series - minimum) / (maximum - minimum)) * 100


# Calculate individual issue scores
df["Volume_Score"] = normalize(df["Review_Count"])

df["Non_Recommendation_Score"] = normalize(
    df["Non_Recommended_Percentage"]
)

# Lower ratings mean greater dissatisfaction
df["Dissatisfaction_Score"] = normalize(
    5 - df["Average_Rating"]
)

# Calculate overall priority score
df["Priority_Score"] = (
    0.4 * df["Volume_Score"]
    + 0.4 * df["Non_Recommendation_Score"]
    + 0.2 * df["Dissatisfaction_Score"]
)

# Rank issues from highest to lowest priority
df = df.sort_values(
    by="Priority_Score",
    ascending=False
).reset_index(drop=True)

df.insert(0, "Priority_Rank", range(1, len(df) + 1))

# Display results
print("\nISSUE PRIORITY RANKING")
print("=" * 50)

print(
    df[
        [
            "Priority_Rank",
            "Topic_Label",
            "Review_Count",
            "Average_Rating",
            "Non_Recommended_Percentage",
            "Priority_Score"
        ]
    ].round(2).to_string(index=False)
)

# Save report
output_file = OUTPUT_DIR / "issue_priority_report.csv"

df.to_csv(output_file, index=False)

print("\nIssue prioritization completed successfully!")
print(f"Report saved to: {output_file}")
