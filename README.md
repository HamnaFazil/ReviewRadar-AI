# ReviewRadar AI – Intelligent Customer Feedback Analysis

An NLP and Machine Learning project designed to analyze customer reviews, identify recurring complaints, discover product-related issues, and generate actionable business insights.

## Project Overview

ReviewRadar AI aims to transform unstructured customer feedback into meaningful business intelligence.

Businesses receive large volumes of customer reviews, making it difficult to manually identify recurring problems and understand customer satisfaction.

This project explores how Natural Language Processing (NLP), Machine Learning, and data visualization can help businesses understand customer feedback more effectively.

### Project Objectives

- Analyze customer reviews and satisfaction levels.
- Identify recurring customer complaints and feedback themes.
- Discover product-related issues using NLP techniques.
- Classify customer sentiment.
- Prioritize important customer issues.
- Generate evidence-based business recommendations.
- Present insights through an interactive dashboard.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning and NLP |
| TF-IDF | Text feature extraction |
| NMF | Topic modelling |
| Logistic Regression | Sentiment classification |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualizations |
| Streamlit | Interactive dashboard |

## Dataset

**Dataset:** Women's E-Commerce Clothing Reviews

The dataset contains customer reviews and product-related information.

### Original Dataset Information

- Total records: 23,486
- Original columns: 11
- Average customer rating: 4.2/5

### Main Dataset Attributes

- Clothing ID
- Age
- Title
- Review Text
- Rating
- Recommended IND
- Positive Feedback Count
- Division Name
- Department Name
- Class Name

## Project Development Progress

### Day 1 – Project Setup and Environment Configuration

**Status: Completed**

Activities:
- Created the ReviewRadar AI project.
- Established the project folder structure.
- Configured the Python development environment.
- Created and activated a Python virtual environment.
- Installed the required Python libraries.
- Prepared the project for data analysis and machine learning development.

**Development environment:**
- Python 3.13.17
- Visual Studio Code
- Windows 11
- Python virtual environment (`.venv`)

### Day 2 – Data Cleaning and Exploratory Data Analysis (EDA)

**Status: Completed**

The second milestone focused on understanding the dataset, examining data quality, cleaning the data, and identifying initial customer satisfaction patterns.

#### 1. Data Inspection

- Loaded 23,486 customer reviews.
- Examined the dataset structure and column information.
- Inspected the first five records.
- Identified missing values.
- Checked duplicate records.

#### 2. Data Cleaning

- Standardized column names.
- Checked and removed duplicate records.
- Converted and examined rating values.
- Created a cleaned review text column.
- Retained records with missing review text for non-text-based analysis.

**Cleaning results:**

| Metric | Result |
|---|---:|
| Original records | 23,486 |
| Records after cleaning | 23,486 |
| Duplicate records removed | 0 |
| Missing review texts | 845 |
| Missing titles | 3,810 |
| Missing department values | 14 |

Missing review texts will be excluded from subsequent NLP processing.

#### 3. Customer Rating Analysis

| Rating | Number of Reviews |
|---|---:|
| 1 Star | 842 |
| 2 Stars | 1,565 |
| 3 Stars | 2,871 |
| 4 Stars | 5,077 |
| 5 Stars | 13,131 |

**Average customer rating: 4.2/5**

The dataset contains a relatively high number of positive customer ratings, with five-star reviews representing the largest group.

#### 4. Product Recommendation Analysis

| Recommendation | Number of Reviews |
|---|---:|
| Recommended | 19,314 |
| Not Recommended | 4,172 |

Approximately 82.2% of the reviews indicate that customers recommended the products.

#### 5. Department-Level Analysis

| Department | Review Count | Average Rating |
|---|---:|---:|
| Bottoms | 3,799 | 4.29 |
| Intimate | 1,735 | 4.28 |
| Jackets | 1,032 | 4.26 |
| Tops | 10,468 | 4.17 |
| Dresses | 6,319 | 4.15 |
| Trend | 119 | 3.82 |

The Trend department recorded the lowest average rating (3.82), while Bottoms recorded the highest average rating (4.29) among the listed departments.

The Trend department also has a considerably smaller review volume, which should be considered when interpreting this result.

#### 6. Visualizations Generated

The following visualizations were generated during EDA:

- Customer rating distribution
- Product recommendation distribution
- Department-wise average ratings
- Product class average ratings

**Visualization files:**

- `rating_distribution.png`
- `recommendation_distribution.png`
- `department_ratings.png`
- `class_ratings.png`

All generated charts are available in `reports/figures/`.

#### 7. Day 2 Deliverables

- Cleaned dataset
- Department summary
- Product class summary
- Data cleaning and EDA Python script
- Four data visualizations

**Output files:**

- `data/processed/cleaned_reviews.csv`
- `data/processed/department_summary.csv`
- `data/processed/class_summary.csv`
- `reports/figures/`

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
│       └── class_summary.csv
│
├── notebooks/
│   └── cleaning_eda.py
│
├── reports/
│   └── figures/
│       ├── rating_distribution.png
│       ├── recommendation_distribution.png
│       ├── department_ratings.png
│       └── class_ratings.png
│
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd ReviewRadar-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit
```

### 6. Add the Dataset

Place the dataset in:

```text
data/raw/clothing_reviews.csv
```

### 7. Run the EDA Script

```bash
python notebooks/cleaning_eda.py
```

## Upcoming Development Milestones

| Milestone | Planned Activities | Status |
|---|---|---|
| Day 1 | Project setup and environment configuration | Completed |
| Day 2 | Data cleaning and exploratory data analysis | Completed |
| Day 3 | Customer complaint and theme analysis | In Progress |
| Day 4 | Sentiment classification and model development | Planned |
| Day 5 | Model evaluation and issue prioritization | Planned |
| Day 6 | Interactive dashboard development | Planned |
| Day 7 | Business recommendations and final documentation | Planned |

## Expected Final Outcome

The final application aims to provide:

- Customer sentiment insights.
- Recurring complaint identification.
- Product issue categorization.
- Evidence-based issue prioritization.
- Representative customer feedback.
- Business improvement recommendations.
- An interactive customer feedback analytics dashboard.

## Project Status

**Currently completed:** Project setup, data cleaning, and exploratory data analysis.

Further NLP, machine learning, and dashboard development will be documented as the project progresses.

---

*ReviewRadar AI – Turning Customer Feedback into Business Intelligence.*