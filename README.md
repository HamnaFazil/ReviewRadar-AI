# ReviewRadar AI – Intelligent Customer Feedback Analysis

An end-to-end Machine Learning and Natural Language Processing (NLP) project that analyzes customer reviews, discovers recurring feedback themes, identifies customer dissatisfaction, and prioritizes product-related issues to support data-driven business decisions.

## Project Overview

ReviewRadar AI transforms unstructured customer reviews into meaningful business insights.

The system aims to help businesses understand:

- What customers appreciate about their products.
- What problems customers frequently mention.
- Which product-related issues require attention.
- How customer satisfaction differs across feedback themes.
- Whether customer reviews express positive, neutral, or negative sentiment.
- Which issues should receive higher business priority.

The project combines Python, Data Analysis, NLP, Machine Learning, and Data Visualization.

## Project Objectives

1. Clean and prepare customer review data.
2. Perform Exploratory Data Analysis (EDA).
3. Discover recurring customer feedback themes using topic modelling.
4. Analyze satisfaction levels across discovered topics.
5. Prioritize customer issues using a weighted scoring approach.
6. Develop a machine learning sentiment classification model.
7. Evaluate model performance using appropriate metrics.
8. Build an interactive dashboard.
9. Generate actionable business recommendations.
10. Present customer feedback insights through a complete analytics solution.

## Dataset

**Dataset:** Women's E-Commerce Clothing Reviews

**Source:** Kaggle

The dataset contains customer reviews and product-related information.

Important attributes include:

- Review Text
- Rating
- Recommended IND
- Department Name
- Class Name
- Title
- Positive Feedback Count

### Dataset Summary

| Metric | Result |
|---|---:|
| Total reviews | 23,486 |
| Original columns | 11 |
| Duplicate rows | 0 |
| Missing review texts | 845 |
| Usable review texts for NLP | 22,641 |

Empty review texts were excluded from NLP processing while retaining the corresponding records for other analyses.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and NLP |
| TF-IDF | Text feature extraction |
| NMF | Topic modelling |
| Logistic Regression | Sentiment classification |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualization |
| Streamlit | Interactive dashboard |
| Joblib | Saving trained machine learning models |

# Project Development – First Four Days

## Day 1 – Project Setup and Environment Configuration

**Objective:** Establish the development environment and organize the project structure.

### Completed activities

- Created the ReviewRadar AI project.
- Configured a Python virtual environment.
- Installed the required Python libraries.
- Organized project directories.
- Created the initial README documentation.
- Configured `.gitignore` to exclude unnecessary files and datasets from Git tracking.

### Environment details

- Python 3.13.17
- Virtual environment: `.venv`
- IDE: Visual Studio Code

### Project paths

| Item | Path |
|---|---|
| Project root | `ReviewRadar-AI/` |
| Virtual environment | `.venv/` |
| Raw dataset folder | `data/raw/` |
| Processed data folder | `data/processed/` |
| Python scripts | `notebooks/` |
| Reports | `reports/` |
| Visualizations | `reports/figures/` |
| Documentation | `README.md` |
| Dependencies | `requirements.txt` |

### Result summary

Successfully configured the Python development environment and prepared the project structure for data analysis and machine learning development.

---

## Day 2 – Data Cleaning and Exploratory Data Analysis

**Objective:** Understand the dataset, clean the data, and identify initial customer satisfaction patterns.

### Completed activities

- Loaded the Women's E-Commerce Clothing Reviews dataset.
- Inspected dataset dimensions and column information.
- Identified missing values.
- Checked duplicate records.
- Prepared cleaned review data.
- Analyzed customer rating distributions.
- Examined product recommendation patterns.
- Compared average ratings across product departments and classes.
- Generated exploratory data visualizations.

### Dataset cleaning results

| Metric | Result |
|---|---:|
| Total records | 23,486 |
| Duplicate records | 0 |
| Missing review texts | 845 |
| Missing titles | 3,810 |
| Missing Division Name | 14 |
| Missing Department Name | 14 |
| Missing Class Name | 14 |
| Records after cleaning | 23,486 |

Missing review texts were retained in the cleaned dataset but excluded from text-based NLP analysis.

### Customer rating distribution

| Rating | Number of Reviews |
|---|---:|
| 1 Star | 842 |
| 2 Stars | 1,565 |
| 3 Stars | 2,871 |
| 4 Stars | 5,077 |
| 5 Stars | 13,131 |

### Key findings

- Overall average rating: **4.2**
- Recommended reviews: **19,314**
- Non-recommended reviews: **4,172**
- Approximately **82.2%** of customers recommended the products.

### Department-level analysis

| Department | Reviews | Average Rating |
|---|---:|---:|
| Bottoms | 3,799 | 4.29 |
| Intimate | 1,735 | 4.28 |
| Jackets | 1,032 | 4.26 |
| Tops | 10,468 | 4.17 |
| Dresses | 6,319 | 4.15 |
| Trend | 119 | 3.82 |

The Trend department had the lowest average rating, although its review volume was relatively small.

### Generated files and report paths

**Python script:**

`notebooks/cleaning_eda.py`

**Processed datasets:**

- `data/processed/cleaned_reviews.csv`
- `data/processed/department_summary.csv`
- `data/processed/class_summary.csv`

**Visualizations:**

- `reports/figures/rating_distribution.png`
- `reports/figures/recommendation_distribution.png`
- `reports/figures/department_ratings.png`
- `reports/figures/class_ratings.png`

### Result summary

Successfully completed data cleaning and EDA, identifying overall customer satisfaction patterns, recommendation distributions, and department-level rating differences.

---

## Day 3 – Customer Feedback Theme Analysis and Issue Prioritization

**Objective:** Discover recurring customer feedback themes using NLP and identify the issues requiring the most business attention.

### 1. Text Processing and Feature Extraction

Applied TF-IDF (Term Frequency–Inverse Document Frequency) to convert customer review text into numerical features.

Configuration:

- Maximum features: 3,000
- N-gram range: (1, 2)
- Minimum document frequency: 5
- Maximum document frequency: 90%
- Stop words: English

**Results:**

| Metric | Result |
|---|---:|
| Total reviews | 23,486 |
| Usable reviews | 22,641 |
| Excluded empty reviews | 845 |
| TF-IDF features | 3,000 |
| TF-IDF matrix | 22,641 × 3,000 |

### 2. Topic Modelling Using NMF

Applied Non-negative Matrix Factorization (NMF) to discover recurring themes in customer reviews.

Model configuration:

- Number of topics: 8
- Random state: 42
- Initialization: `nndsvda`
- Maximum iterations: 400

Eight topics were discovered and assigned descriptive business labels through manual interpretation of their keywords and representative reviews.

### Discovered customer feedback themes

| Topic | Theme | Review Count | Percentage |
|---|---|---:|---:|
| 1 | General Product Feedback | 3,975 | 17.56% |
| 2 | Dress Design & Appearance | 3,645 | 16.10% |
| 3 | Comfort & Wearability | 4,445 | 19.63% |
| 4 | Size Accuracy & Fit | 2,078 | 9.18% |
| 5 | Sizing & Size Selection | 3,168 | 13.99% |
| 6 | Shirt Design & Material | 1,551 | 6.85% |
| 7 | Sweater & Outerwear Feedback | 1,706 | 7.54% |
| 8 | Product Quality & Appearance | 2,073 | 9.16% |
| **Total** | **8 Topics** | **22,641** | **100%** |

### 3. Topic Satisfaction Analysis

Analyzed average customer ratings and non-recommendation percentages for each discovered topic.

| Customer Feedback Theme | Average Rating | Non-Recommended Count | Non-Recommended % |
|---|---:|---:|---:|
| General Product Feedback | 3.59 | 1,440 | 36.23% |
| Dress Design & Appearance | 4.14 | 728 | 19.97% |
| Comfort & Wearability | 4.64 | 274 | 6.16% |
| Size Accuracy & Fit | 4.42 | 208 | 10.01% |
| Sizing & Size Selection | 4.06 | 598 | 18.88% |
| Shirt Design & Material | 3.97 | 371 | 23.92% |
| Sweater & Outerwear Feedback | 4.20 | 321 | 18.82% |
| Product Quality & Appearance | 4.53 | 161 | 7.77% |

### Key findings

- General Product Feedback has the highest non-recommendation percentage (36.23%) and lowest average rating (3.59).
- Shirt Design & Material has a 23.92% non-recommendation rate.
- Sizing & Size Selection also indicates potential customer dissatisfaction.
- Comfort & Wearability has the highest average rating (4.64).
- Product Quality & Appearance has a relatively low non-recommendation rate (7.77%).

### 4. Representative Review Extraction

Extracted representative reviews for each discovered topic.

This provides qualitative evidence to help interpret the numerical topic analysis and understand the customer experiences behind each theme.

### 5. Issue Prioritization

Developed a weighted scoring method to rank customer feedback themes based on:

- Review volume
- Non-recommendation percentage
- Rating dissatisfaction

**Project-defined priority formula:**

`Priority Score = (0.4 × V) + (0.4 × N) + (0.2 × D)`

Where:

- V = Normalized review volume (0–100)
- N = Non-recommendation percentage (0–100)
- D = Normalized rating dissatisfaction (0–100)

### Issue priority ranking

| Rank | Customer Feedback Theme | Priority Score |
|---:|---|---:|
| 1 | General Product Feedback | 93.50 |
| 2 | Dress Design & Appearance | 56.84 |
| 3 | Sizing & Size Selection | 50.32 |
| 4 | Comfort & Wearability | 40.00 |
| 5 | Shirt Design & Material | 36.39 |
| 6 | Sweater & Outerwear Feedback | 27.36 |
| 7 | Size Accuracy & Fit | 16.60 |
| 8 | Product Quality & Appearance | 11.45 |

### Business interpretation

General Product Feedback received the highest priority score. Representative reviews indicate complaints involving fabric quality, product fit, and value for money.

Dress Design & Appearance and Sizing & Size Selection also deserve attention based on their review volumes and dissatisfaction indicators.

This score is a project-defined decision-support heuristic, not a statistically validated business-risk measurement. The results identify associations and potential areas for investigation rather than proving the causes of dissatisfaction.

### Generated files and report paths

**Python scripts:**

- `notebooks/theme_analysis.py`
- `notebooks/issue_prioritization.py`
- `notebooks/visualize_priorities.py`

**Processed datasets and analysis reports:**

- `data/processed/topic_summary.csv`
- `data/processed/review_topic_assignments.csv`
- `data/processed/representative_reviews.csv`
- `data/processed/topic_satisfaction.csv`
- `data/processed/issue_priority_report.csv`

**Visualization:**

- `reports/figures/issue_priority.png`

### Result summary

Successfully discovered eight customer feedback themes using NMF, analyzed satisfaction across the topics, extracted representative reviews, and generated a weighted issue-priority ranking with a corresponding visualization.

---

## Day 4 – Sentiment Classification Using Machine Learning

**Objective:** Develop a machine learning model that classifies customer reviews into positive, neutral, and negative sentiment categories.

### 1. Sentiment Label Creation

The dataset does not contain a manually verified sentiment label. Therefore, customer ratings are used to create rating-based sentiment categories.

| Rating | Sentiment |
|---|---|
| 1–2 Stars | Negative |
| 3 Stars | Neutral |
| 4–5 Stars | Positive |

These labels represent rating-based sentiment proxies rather than manually annotated sentiment.

### 2. Text Preparation

- Loaded the cleaned customer review dataset.
- Selected `Review_Text_Clean` and `Rating`.
- Removed missing ratings and empty review texts.
- Created sentiment labels from customer ratings.
- Prepared the text features and target labels for classification.

### 3. Training and Testing

Used an 80:20 training/testing split.

| Dataset | Proportion |
|---|---:|
| Training data | 80% |
| Testing data | 20% |

Used stratified splitting to preserve sentiment class proportions across the training and testing sets.

### 4. Machine Learning Model

**TF-IDF + Logistic Regression**

TF-IDF converts review text into numerical features, while Logistic Regression learns patterns that help classify the reviews into sentiment categories.

Model configuration:

| Parameter | Value |
|---|---|
| Maximum TF-IDF features | 5,000 |
| N-gram range | (1, 2) |
| Minimum document frequency | 3 |
| Maximum document frequency | 90% |
| Class weighting | Balanced |
| Maximum iterations | 1,000 |
| Random state | 42 |

A Scikit-learn pipeline keeps text vectorization and model training together. The TF-IDF vocabulary is learned from training data only.

### 5. Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics help evaluate the model's ability to recognize positive, neutral, and negative customer reviews.

**Execution status:** The sentiment classification script has been prepared. Actual model performance metrics will be added after successful execution and evaluation.

### Generated files and report paths

**Python script:**

- `notebooks/sentiment_classification.py`

**Trained model:**

- `models/sentiment_model.pkl`

**Evaluation reports:**

- `reports/sentiment_classification_report.csv`
- `reports/sentiment_metrics.json`

**Visualization:**

- `reports/figures/sentiment_confusion_matrix.png`

These output files are generated when the sentiment classification script runs successfully.

### Result summary

Implemented a rating-based sentiment classification workflow using TF-IDF and Logistic Regression. The workflow includes stratified train/test splitting, model training, classification metrics, confusion matrix generation, and model serialization.

Actual evaluation results are pending execution.

---

# Project Directory Structure

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
├── models/
│   └── sentiment_model.pkl
│
├── notebooks/
│   ├── cleaning_eda.py
│   ├── theme_analysis.py
│   ├── issue_prioritization.py
│   ├── visualize_priorities.py
│   └── sentiment_classification.py
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
├── .gitignore
├── README.md
└── requirements.txt
```

Note: Some Day 4 output files are created only after the classification script runs successfully. Raw and processed CSV datasets are excluded from Git tracking according to the project's `.gitignore` rules.

# How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd ReviewRadar-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the analysis scripts

Data cleaning and EDA:

```bash
python notebooks/cleaning_eda.py
```

Topic modelling:

```bash
python notebooks/theme_analysis.py
```

Issue prioritization:

```bash
python notebooks/issue_prioritization.py
```

Priority visualization:

```bash
python notebooks/visualize_priorities.py
```

Sentiment classification:

```bash
python notebooks/sentiment_classification.py
```

Make sure the original dataset is available in `data/raw/` before running the scripts.

# Project Limitations

- Topic labels are manually interpreted descriptions of NMF-discovered topics.
- Sentiment labels are derived from ratings and are not manually annotated.
- The issue priority score is a project-defined heuristic.
- Review themes and satisfaction measurements do not establish causal relationships.
- The current dataset does not support time-based trend analysis in this project.
- Model performance must be assessed using multiple evaluation metrics, not accuracy alone.

# Author

**Hamna Fazil**

Aspiring Data Analyst | Business Intelligence | Machine Learning

GitHub: [HamnaFazil](https://github.com/HamnaFazil)

Portfolio: [Personal Portfolio](https://hamnafazil.github.io/HamnaFazil/)

---

*ReviewRadar AI – Turning customer feedback into actionable business intelligence.*