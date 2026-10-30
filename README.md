# ReviewRadar AI – Intelligent Customer Feedback Analysis

An end-to-end Machine Learning and Natural Language Processing project that analyzes customer reviews, discovers recurring feedback themes, identifies customer dissatisfaction, and prioritizes product-related issues to support data-driven business decisions.

## Project Overview

ReviewRadar AI transforms unstructured customer reviews into meaningful business insights.

The system aims to help businesses understand:
- What customers appreciate about their products.
- What problems customers frequently mention.
- Which product-related issues require attention.
- How customer satisfaction differs across feedback themes.
- Which issues should receive higher business priority.

The project combines data analysis, Natural Language Processing (NLP), Machine Learning, and interactive visualization.

## Objectives

1. Clean and prepare customer review data.
2. Perform Exploratory Data Analysis (EDA).
3. Discover recurring customer feedback themes using topic modelling.
4. Analyze satisfaction levels across discovered topics.
5. Prioritize customer issues using a weighted scoring approach.
6. Develop a machine learning model for sentiment classification.
7. Build an interactive dashboard to communicate business insights.
8. Generate actionable business recommendations.

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
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualizations |
| Streamlit | Interactive dashboard |

## Project Progress

### Day 1 – Project Setup

- Created the ReviewRadar AI project structure.
- Configured the Python virtual environment.
- Installed the required Python libraries.
- Organized project directories for data, notebooks, and reports.
- Created the initial README and `.gitignore`.

### Day 2 – Data Cleaning and Exploratory Data Analysis

Performed data cleaning and exploratory analysis on the Women's E-Commerce Clothing Reviews dataset.

**Key findings:**

- Analyzed 23,486 customer reviews.
- Identified 845 missing review texts.
- Found no duplicate rows.
- Prepared the cleaned dataset for further analysis.
- Examined customer ratings and recommendation patterns.
- Analyzed average ratings across product departments.
- Generated visualizations for rating and recommendation distributions.

**Rating distribution:**

| Rating | Number of Reviews |
|---|---:|
| 1 Star | 842 |
| 2 Stars | 1,565 |
| 3 Stars | 2,871 |
| 4 Stars | 5,077 |
| 5 Stars | 13,131 |

**Additional findings:**

- Overall average rating: 4.2
- Recommended reviews: 19,314
- Non-recommended reviews: 4,172
- Approximately 82.2% of reviews recommended the products.

**Generated outputs:**
- `cleaned_reviews.csv`
- `department_summary.csv`
- `class_summary.csv`
- `rating_distribution.png`
- `recommendation_distribution.png`
- `department_ratings.png`
- `class_ratings.png`

### Day 3 – Customer Feedback Theme Analysis and Issue Prioritization

Applied Natural Language Processing and Non-negative Matrix Factorization (NMF) to discover recurring themes in customer reviews.

**1. Text Processing and Feature Extraction**

- Prepared customer review text for NLP analysis.
- Excluded 845 empty review texts.
- Used TF-IDF Vectorization to convert review text into numerical features.
- Configured the vectorizer with a maximum of 3,000 features and unigrams and bigrams.
- Generated a TF-IDF matrix with 22,641 reviews and 3,000 features.

**2. Topic Modelling**

Applied NMF with eight components to identify recurring customer feedback themes.

Eight topics were discovered and manually assigned business-friendly labels after reviewing their keywords and representative reviews.

| Topic | Discovered Theme | Reviews | Percentage |
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

**3. Topic Satisfaction Analysis**

Analyzed average ratings and non-recommendation percentages across the eight discovered topics.

| Customer Feedback Theme | Average Rating | Non-Recommended |
|---|---:|---:|
| General Product Feedback | 3.59 | 36.23% |
| Dress Design & Appearance | 4.14 | 19.97% |
| Comfort & Wearability | 4.64 | 6.16% |
| Size Accuracy & Fit | 4.42 | 10.01% |
| Sizing & Size Selection | 4.06 | 18.88% |
| Shirt Design & Material | 3.97 | 23.92% |
| Sweater & Outerwear Feedback | 4.20 | 18.82% |
| Product Quality & Appearance | 4.53 | 7.77% |

**Key insights:**

- General Product Feedback showed the highest dissatisfaction, with a 36.23% non-recommendation rate and a 3.59 average rating.
- Shirt Design & Material also showed an area of concern, with a 23.92% non-recommendation rate.
- Sizing & Size Selection showed potential sizing-related customer dissatisfaction.
- Comfort & Wearability performed positively, with a 4.64 average rating and only 6.16% non-recommendation.

**4. Representative Review Extraction**

Extracted representative customer reviews for each discovered topic to provide qualitative evidence behind the topic classifications.

This helps connect numerical analysis with actual customer experiences.

**5. Issue Prioritization**

Developed a weighted issue-prioritization approach using three factors:

- Review Volume
- Non-Recommended Percentage
- Rating Dissatisfaction

The project-defined priority formula is:

`Priority Score = (0.4 × V) + (0.4 × N) + (0.2 × D)`

Where:

- V = Normalized review volume
- N = Non-recommendation percentage
- D = Normalized rating dissatisfaction

Each factor is normalized to a 0–100 scale before calculating the final score.

**Issue Priority Ranking:**

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

**Business interpretation:**

General Product Feedback received the highest priority score. Its high dissatisfaction level indicates that the business should investigate the specific complaints represented in this topic, such as fabric quality, product fit, and value for money.

This prioritization method is a project-defined decision-support heuristic and does not represent a statistically validated business-risk score.

**Generated Day 3 outputs:**
- `topic_summary.csv`
- `review_topic_assignments.csv`
- `representative_reviews.csv`
- `topic_satisfaction.csv`
- `issue_priority_report.csv`

## Project Structure

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
│   └── issue_prioritization.py
│
├── reports/
│   └── figures/
│
├── .gitignore
├── README.md
└── requirements.txt
```

Note: Raw and processed CSV files are excluded from Git tracking according to the project's `.gitignore` rules.

## How to Run the Project

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

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the analysis scripts

```bash
python notebooks/cleaning_eda.py
```

```bash
python notebooks/theme_analysis.py
```

```bash
python notebooks/issue_prioritization.py
```

Make sure the original dataset is placed inside `data/raw/` before running the scripts.

## Upcoming Development

### Day 4 – Sentiment Classification
- Prepare labelled sentiment data.
- Develop a sentiment classification model.
- Train and test the model.
- Evaluate model performance using suitable classification metrics.

### Day 5 – Model Evaluation and Improvement
- Analyze model performance.
- Examine misclassified reviews.
- Improve model reliability.
- Compare relevant evaluation metrics.

### Day 6 – Interactive Dashboard
- Develop an interactive Streamlit dashboard.
- Display customer satisfaction insights.
- Visualize discovered topics and issue priorities.
- Allow users to explore customer feedback.

### Day 7 – Business Recommendations and Finalization
- Generate actionable business recommendations.
- Complete project documentation.
- Prepare the final GitHub portfolio presentation.

## Limitations

- Topic labels are manually interpreted descriptions of the NMF-discovered topics.
- The priority score is a project-defined heuristic.
- Review ratings and recommendation indicators are used as satisfaction signals.
- Topic associations do not establish the actual causes of customer dissatisfaction.
- The dataset does not currently support time-based trend analysis in this project.

## Project Status

**Current progress: Day 3 completed**

- [x] Project setup
- [x] Data cleaning
- [x] Exploratory Data Analysis
- [x] TF-IDF feature extraction
- [x] NMF topic modelling
- [x] Topic satisfaction analysis
- [x] Representative review extraction
- [x] Issue prioritization
- [ ] Sentiment classification
- [ ] Model evaluation
- [ ] Interactive dashboard
- [ ] Business recommendations
- [ ] Final documentation

## Author

**Hamna Fazil**

Aspiring Data Analyst | Business Intelligence | Machine Learning

GitHub: [HamnaFazil](https://github.com/HamnaFazil)

Portfolio: [Personal Portfolio](https://hamnafazil.github.io/HamnaFazil/)

---

*ReviewRadar AI – Turning customer feedback into actionable business intelligence.*