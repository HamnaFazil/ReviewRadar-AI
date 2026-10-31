# ReviewRadar AI – Intelligent Customer Feedback Analysis

## Project Overview

ReviewRadar AI is a machine learning and natural language processing project designed to analyze customer reviews, identify recurring feedback themes, classify customer sentiment, and prioritize potential business improvement areas.

The project uses the Women's E-Commerce Clothing Reviews dataset to extract meaningful insights from customer feedback and present them in a business-friendly format.

## Project Objectives

- Analyze customer ratings and feedback.
- Clean and prepare customer review data.
- Identify recurring topics using NLP and topic modelling.
- Measure satisfaction levels across identified topics.
- Prioritize issues based on review volume and dissatisfaction.
- Classify reviews into positive, neutral, and negative sentiment.
- Generate visual reports to support business decisions.

## Dataset

**Dataset:** Women's E-Commerce Clothing Reviews

**Original file:** `data/raw/clothing_reviews.csv`

**Original records:** 23,486 reviews

**Usable text reviews:** 22,641

**Empty review texts:** 845

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Non-negative Matrix Factorization (NMF)
- Logistic Regression
- Matplotlib
- Seaborn
- Joblib
- Git and GitHub

---

# Development Progress

## Day 1 – Project Setup and Environment Configuration

### Completed Tasks

- Created the ReviewRadar AI project structure.
- Configured the Python virtual environment.
- Installed and verified the required Python libraries.
- Downloaded and organized the dataset.
- Created folders for notebooks, raw data, processed data, reports, and visualizations.
- Created the initial GitHub repository structure.

### Project Structure

```text
ReviewRadar-AI/
│
├── data/
│   ├── raw/
│   │   └── clothing_reviews.csv
│   │
│   └── processed/
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── models/
│
├── README.md
├── .gitignore
└── .venv/
```

### Result

The development environment was successfully configured, and the project was ready for data analysis.

---

## Day 2 – Data Cleaning and Exploratory Data Analysis

### Completed Tasks

- Loaded the original dataset using Pandas.
- Inspected dataset structure and data types.
- Identified missing values.
- Checked duplicate records.
- Cleaned and prepared review text.
- Analyzed customer rating distributions.
- Analyzed recommendation patterns.
- Compared customer ratings across product departments and classes.
- Generated exploratory data visualizations.

### Results Summary

| Analysis | Result |
|---|---:|
| Total reviews | 23,486 |
| Original columns | 11 |
| Duplicate rows | 0 |
| Missing review texts | 845 |
| Missing titles | 3,810 |
| Missing department values | 14 |
| Average rating | 4.2 |
| Recommended reviews | 19,314 |
| Non-recommended reviews | 4,172 |

Approximately 82.2% of customers recommended the products.

### Average Rating by Department

| Department | Average Rating |
|---|---:|
| Bottoms | 4.29 |
| Intimate | 4.28 |
| Jackets | 4.26 |
| Tops | 4.17 |
| Dresses | 4.15 |
| Trend | 3.82 |

The Trend department had the lowest average rating, but it also had a relatively small number of reviews (119).

### Generated Reports

**Processed datasets:**

```text
data/processed/cleaned_reviews.csv
data/processed/department_summary.csv
data/processed/class_summary.csv
```

**Visualizations:**

```text
reports/figures/rating_distribution.png
reports/figures/recommendation_distribution.png
reports/figures/department_ratings.png
reports/figures/class_ratings.png
```

### Result

The dataset was successfully cleaned and explored. Initial customer satisfaction patterns and differences between product departments were identified.

---

## Day 3 – Topic Modelling and Issue Prioritization

### Completed Tasks

- Applied TF-IDF vectorization to customer review text.
- Used Non-negative Matrix Factorization (NMF) for topic modelling.
- Identified eight recurring customer feedback topics.
- Extracted representative keywords for each topic.
- Assigned reviews to their dominant topics.
- Interpreted and manually labelled the identified topics.
- Analyzed average ratings and non-recommendation percentages by topic.
- Developed a project-defined issue prioritization score.
- Generated an issue priority visualization.

### Topic Modelling Configuration

| Parameter | Value |
|---|---|
| Algorithm | NMF |
| Vectorization | TF-IDF |
| Maximum features | 3,000 |
| Number of topics | 8 |
| N-gram range | (1, 2) |
| Random state | 42 |
| Maximum iterations | 400 |

### Identified Customer Feedback Topics

| Topic ID | Topic Label | Reviews | Percentage |
|---|---|---:|---:|
| 1 | General Product Feedback | 3,975 | 17.56% |
| 2 | Dress Design & Appearance | 3,645 | 16.10% |
| 3 | Comfort & Wearability | 4,445 | 19.63% |
| 4 | Size Accuracy & Fit | 2,078 | 9.18% |
| 5 | Sizing & Size Selection | 3,168 | 13.99% |
| 6 | Shirt Design & Material | 1,551 | 6.85% |
| 7 | Sweater & Outerwear Feedback | 1,706 | 7.54% |
| 8 | Product Quality & Appearance | 2,073 | 9.16% |

All 22,641 usable text reviews were assigned to a topic.

### Topic Satisfaction Analysis

| Topic | Average Rating | Non-recommended |
|---|---:|---:|
| General Product Feedback | 3.59 | 36.23% |
| Dress Design & Appearance | 4.14 | 19.97% |
| Comfort & Wearability | 4.64 | 6.16% |
| Size Accuracy & Fit | 4.42 | 10.01% |
| Sizing & Size Selection | 4.06 | 18.88% |
| Shirt Design & Material | 3.97 | 23.92% |
| Sweater & Outerwear Feedback | 4.20 | 18.82% |
| Product Quality & Appearance | 4.53 | 7.77% |

### Key Findings

- General Product Feedback showed the highest dissatisfaction, with a 36.23% non-recommendation rate.
- Shirt Design & Material also showed an area of concern, with a 23.92% non-recommendation rate.
- Sizing & Size Selection had an 18.88% non-recommendation rate.
- Comfort & Wearability received the highest average rating of 4.64.
- Product Quality & Appearance also received a relatively high average rating of 4.53.

### Issue Prioritization

A project-defined priority score was developed using:

```text
Priority Score = 0.4V + 0.4N + 0.2D
```

Where:

- V = Normalized review volume.
- N = Non-recommendation percentage.
- D = Normalized rating dissatisfaction.

### Issue Priority Ranking

| Rank | Topic | Priority Score |
|---|---|---:|
| 1 | General Product Feedback | 93.50 |
| 2 | Dress Design & Appearance | 56.84 |
| 3 | Sizing & Size Selection | 50.32 |
| 4 | Comfort & Wearability | 40.00 |
| 5 | Shirt Design & Material | 36.39 |
| 6 | Sweater & Outerwear Feedback | 27.36 |
| 7 | Size Accuracy & Fit | 16.60 |
| 8 | Product Quality & Appearance | 11.45 |

The score is a project-defined prioritization heuristic, not a validated measure of business risk.

### Generated Reports

**Processed datasets:**

```text
data/processed/topic_summary.csv
data/processed/review_topic_assignments.csv
data/processed/representative_reviews.csv
data/processed/topic_satisfaction.csv
data/processed/issue_priority_report.csv
```

**Visualization:**

```text
reports/figures/issue_priority.png
```

### Result

Eight recurring customer feedback topics were identified, evaluated, and prioritized. The analysis provides a structured way to identify areas that may require business attention.

---

## Day 4 – Customer Sentiment Classification

### Completed Tasks

- Prepared customer review text for sentiment classification.
- Converted product ratings into sentiment categories.
- Split the dataset into training and testing sets.
- Applied TF-IDF vectorization.
- Trained a Logistic Regression classification model.
- Evaluated model performance using accuracy, precision, recall, and F1-score.
- Generated a confusion matrix.
- Saved the trained model and evaluation reports.

### Sentiment Labelling Strategy

Customer ratings were converted into three sentiment categories:

| Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

These are rating-derived sentiment labels rather than manually annotated sentiment categories.

### Dataset Distribution

| Sentiment | Number of Reviews |
|---|---:|
| Positive | 17,448 |
| Neutral | 2,823 |
| Negative | 2,370 |
| **Total** | **22,641** |

### Model Configuration

| Component | Configuration |
|---|---|
| Feature extraction | TF-IDF |
| Maximum features | 5,000 |
| N-gram range | (1, 2) |
| Classifier | Logistic Regression |
| Maximum iterations | 1,000 |
| Class weighting | Balanced |
| Training/testing split | 80% / 20% |
| Random state | 42 |

### Model Evaluation Results

<details>
<summary>Actual evaluation results</summary>

**Accuracy: 76.97%**

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.47 | 0.60 | 0.53 | 474 |
| Neutral | 0.32 | 0.47 | 0.38 | 565 |
| Positive | 0.95 | 0.84 | 0.89 | 3,490 |
| **Macro Average** | **0.58** | **0.64** | **0.60** | **4,529** |
| **Weighted Average** | **0.82** | **0.77** | **0.79** | **4,529** |

</details>

### Key Findings

- The model achieved an overall accuracy of 76.97%.
- Positive sentiment was classified most effectively, with an F1-score of 0.89.
- Negative sentiment achieved an F1-score of 0.53.
- Neutral sentiment was the most challenging category, with an F1-score of 0.38.
- The difference in sentiment-class sizes indicates an imbalanced dataset.

The results suggest that the model performs better on positive reviews than on neutral and negative reviews. Further model improvement and evaluation are recommended, especially for the minority sentiment categories.

### Generated Reports

**Trained model:**

```text
models/sentiment_model.pkl
```

**Evaluation reports:**

```text
reports/sentiment_classification_report.csv
reports/sentiment_metrics.json
```

**Visualization:**

```text
reports/figures/sentiment_confusion_matrix.png
```

### Result

The TF-IDF and Logistic Regression sentiment classification pipeline was successfully trained and evaluated. The trained model and evaluation outputs were saved for future use in the ReviewRadar AI application.

---

# Complete Project Structure

```text
ReviewRadar-AI/
│
├── data/
│   ├── raw/
│   │   └── clothing_reviews.csv
│   │
│   └── processed/
│       ├── cleaned_reviews.csv
│       ├── department_summary.csv
│       ├── class_summary.csv
│       ├── topic_summary.csv
│       ├── review_topic_assignments.csv
│       ├── representative_reviews.csv
│       ├── topic_satisfaction.csv
│       └── issue_priority_report.csv
│
├── notebooks/
│   ├── cleaning_eda.py
│   ├── theme_analysis.py
│   ├── issue_prioritization.py
│   ├── visualize_priorities.py
│   └── sentiment_classification.py
│
├── models/
│   └── sentiment_model.pkl
│
├── reports/
│   ├── sentiment_classification_report.csv
│   ├── sentiment_metrics.json
│   │
│   └── figures/
│       ├── rating_distribution.png
│       ├── recommendation_distribution.png
│       ├── department_ratings.png
│       ├── class_ratings.png
│       ├── issue_priority.png
│       └── sentiment_confusion_matrix.png
│
├── README.md
└── .gitignore
```

# How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ReviewRadar-AI
```

### 2. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly joblib
```

### 4. Run the analysis scripts

```bash
python notebooks/cleaning_eda.py
python notebooks/theme_analysis.py
python notebooks/issue_prioritization.py
python notebooks/visualize_priorities.py
python notebooks/sentiment_classification.py
```

# Current Project Status

| Development Stage | Status |
|---|---|
| Day 1 – Project Setup | Completed |
| Day 2 – Data Cleaning & EDA | Completed |
| Day 3 – Topic Modelling & Issue Prioritization | Completed |
| Day 4 – Sentiment Classification | Completed |

# Limitations and Future Improvements

- Sentiment labels are derived from star ratings and are not manually verified.
- The dataset contains more positive reviews than negative and neutral reviews.
- Neutral and negative sentiment classification requires further improvement.
- Topic labels were interpreted and assigned manually after NMF topic extraction.
- The priority score is a project-defined heuristic.
- The available analysis does not establish causal relationships or time-based trends.

# Author

**Hamna Fazil**

IT Undergraduate | Data Analytics | Machine Learning | Business Intelligence

GitHub: https://github.com/HamnaFazil

Portfolio: https://hamnafazil.github.io/HamnaFazil/