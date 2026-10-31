
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "issue_priority_report.csv"
OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"

# Create output directory if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load issue priority report
print("Loading issue priority report...")

df = pd.read_csv(INPUT_FILE)

# Sort issues by priority score
df = df.sort_values("Priority_Score", ascending=True)

# Create horizontal bar chart
plt.figure(figsize=(12, 7))

bars = plt.barh(
    df["Topic_Label"],
    df["Priority_Score"]
)

# Add score labels to each bar
for bar in bars:
    width = bar.get_width()

    plt.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        va="center",
        fontsize=10
    )

# Chart formatting
plt.title(
    "ReviewRadar AI - Customer Issue Priority Ranking",
    fontsize=16,
    fontweight="bold",
    pad=20
)

plt.xlabel("Priority Score (0–100)", fontsize=11)
plt.ylabel("Customer Feedback Themes", fontsize=11)

plt.xlim(0, df["Priority_Score"].max() + 15)
plt.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()

# Save chart
OUTPUT_FILE = OUTPUT_DIR / "issue_priority.png"

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPriority visualization created successfully!")
print(f"Chart saved to: {OUTPUT_FILE}")

