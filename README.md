# ReviewRadar AI

### Intelligent Customer Feedback Analysis

ReviewRadar AI is a machine-learning and business-intelligence project
that analyses customer reviews to identify recurring feedback themes,
understand rating-based sentiment, prioritise potential
customer-experience issues, and present findings through an interactive
dashboard.

The project uses the Women's E-Commerce Clothing Reviews dataset. It
combines exploratory data analysis, natural-language processing,
sentiment classification, issue prioritisation, and a Streamlit
dashboard.

> **Project status:** Phases 1--4 are completed. Model evaluation, final
> documentation, and public deployment are planned next.

------------------------------------------------------------------------

## Table of Contents

-   [Project Objectives](#project-objectives)
-   [Key Features](#key-features)
-   [Technology Stack](#technology-stack)
-   [Dataset](#dataset)
-   [Project Development Phases](#project-development-phases)
-   [Project Structure](#project-structure)
-   [Getting Started](#getting-started)
-   [Dashboard Pages](#dashboard-pages)
-   [Methodology and Important Notes](#methodology-and-important-notes)
-   [Current Results](#current-results)
-   [Limitations and Future Work](#limitations-and-future-work)

------------------------------------------------------------------------

## Project Objectives

-   Analyse customer ratings and recommendation behaviour.
-   Identify recurring themes in written customer feedback.
-   Highlight topics that may need further business attention.
-   Classify review sentiment using machine-learning models.
-   Present findings in an accessible interactive dashboard.
-   Support evidence-based customer-experience discussions.

## Key Features

-   Data cleaning and exploratory data analysis (EDA).
-   TF-IDF feature extraction and NMF topic modelling.
-   Topic summaries and representative customer reviews.
-   A project-defined issue-prioritisation heuristic.
-   Rating-derived sentiment labels.
-   Comparison of multiple machine-learning classifiers.
-   Interactive Streamlit dashboard with review exploration and
    sentiment prediction.

## Technology Stack

  -----------------------------------------------------------------------
  Area                                Tools
  ----------------------------------- -----------------------------------
  Programming language                Python

  Data processing                     Pandas, NumPy

  Natural-language processing         Scikit-learn, TF-IDF

  Topic modelling                     Non-negative Matrix Factorization
                                      (NMF)

  Machine learning                    Logistic Regression, Linear SVC,
                                      Complement Naive Bayes

  Visualisation                       Matplotlib, Seaborn, Plotly

  Dashboard                           Streamlit

  Model persistence                   Pickle

  Development tools                   Visual Studio Code, Git, GitHub
  -----------------------------------------------------------------------

## Dataset

**Women's E-Commerce Clothing Reviews**

The dataset contains customer reviews and associated information such as
ratings, recommendation indicators, department, class, age, and positive
feedback counts.

Expected local dataset path:

``` text
data/raw/clothing_reviews.csv
```

The raw dataset is not included in the repository. Place the downloaded
dataset at the path above before running the analysis scripts.

### Dataset preparation summary

-   Original records: 23,486
-   Original columns: 11
-   Missing review texts: 845
-   Missing titles: 3,810
-   Missing values in Division, Department, and Class: 14 in each column
-   Duplicate records identified: 0
-   Reviews with usable text for NLP: 22,641

Empty review texts were excluded from text-based NLP processing. They
were retained in the dataset for analyses that do not require review
text.

------------------------------------------------------------------------

## Project Development Phases

The project is organised into phases rather than individual days. This
reflects the fact that related tasks may be completed together across
different working sessions.

### Phase 1 --- Project Foundations and Data Preparation

**Status: Completed**

This phase established the project environment and prepared the dataset
for analysis.

Tasks completed: - Defined the project purpose and initial workflow. -
Created the project directory structure. - Set up a Python virtual
environment. - Installed and verified the required Python packages. -
Downloaded and organised the e-commerce reviews dataset. - Inspected the
dataset structure, data types, missing values, and duplicates. - Cleaned
the review data and prepared the processed dataset. - Performed
exploratory data analysis (EDA). - Generated rating, recommendation,
department, and class summaries. - Created initial EDA visualisations.

Main outputs: - `data/processed/cleaned_reviews.csv` -
`data/processed/department_summary.csv` -
`data/processed/class_summary.csv` -
`reports/figures/rating_distribution.png` -
`reports/figures/recommendation_distribution.png` -
`reports/figures/department_ratings.png` -
`reports/figures/class_ratings.png`

### Phase 2 --- NLP Theme Discovery and Issue Prioritisation

**Status: Completed**

This phase explored the written feedback to identify recurring topics
and potential areas of customer dissatisfaction.

Tasks completed: - Prepared review text for NLP analysis. - Converted
text into TF-IDF features using unigrams and bigrams. - Applied
Non-negative Matrix Factorization (NMF) topic modelling. - Generated
eight review topics. - Reviewed topic keywords and assigned descriptive
topic labels. - Calculated topic volumes and proportions. - Analysed
average ratings and non-recommendation percentages by topic. - Selected
representative reviews to provide examples of each topic. - Created a
project-defined issue-prioritisation score. - Generated a visualisation
of topic priority scores.

Main outputs: - `data/processed/topic_summary.csv` -
`data/processed/review_topic_assignments.csv` -
`data/processed/representative_reviews.csv` -
`data/processed/topic_satisfaction.csv` -
`data/processed/issue_priority_report.csv` -
`reports/figures/issue_priority.png`

#### Topic modelling configuration

-   TF-IDF maximum features: 3,000
-   N-gram range: (1, 2)
-   Minimum document frequency: 5
-   Maximum document frequency: 0.90
-   NMF components: 8
-   Random state: 42
-   Maximum iterations: 400

#### Issue-prioritisation approach

A project-defined heuristic was used:

`Priority = 0.4V + 0.4N + 0.2D`

Where: - `V` = normalised review volume - `N` = non-recommendation
percentage - `D` = normalised rating dissatisfaction, calculated from
the difference between 5 and the average rating

The components are scaled to a 0--100 range before combining them. This
score is an exploratory prioritisation heuristic created for this
project; it is not a validated business-risk measure.

### Phase 3 --- Sentiment Classification and Model Comparison

**Status: Completed**

This phase developed and compared machine-learning models for review
sentiment classification.

Tasks completed: - Created sentiment labels from star ratings: - Ratings
1--2: Negative - Rating 3: Neutral - Ratings 4--5: Positive - Examined
the class distribution. - Split the data into training and testing sets
using an 80/20 split. - Used stratified sampling and a fixed random
state. - Extracted TF-IDF text features. - Trained a Logistic Regression
baseline classifier. - Evaluated the baseline using accuracy, precision,
recall, F1-score, and a confusion matrix. - Compared Logistic
Regression, Linear Support Vector Classifier (Linear SVC), and
Complement Naive Bayes. - Selected Complement Naive Bayes based on macro
F1-score. - Saved the selected model and evaluation outputs.

Main outputs: - `models/sentiment_model.pkl` -
`models/best_sentiment_model_day5.pkl` -
`reports/sentiment_classification_report.csv` -
`reports/sentiment_metrics.json` - `reports/model_comparison.csv` -
`reports/model_classification_details.csv` -
`reports/best_model_metrics.json` -
`reports/figures/sentiment_confusion_matrix.png` -
`reports/figures/model_comparison.png` -
`reports/figures/best_model_confusion_matrix.png`

#### Baseline model configuration

-   Model: Logistic Regression
-   TF-IDF maximum features: 5,000
-   N-gram range: (1, 2)
-   Minimum document frequency: 3
-   Maximum document frequency: 0.90
-   Class weighting: balanced
-   Maximum iterations: 1,000
-   Train/test split: 80/20
-   Random state: 42

The baseline Logistic Regression model achieved **76.97% accuracy**. Its
macro F1-score was **0.60**. Performance varied across classes, with
lower results for neutral and negative reviews than for positive
reviews.

### Phase 4 --- Interactive Dashboard Development

**Status: Completed**

This phase brought the analysis and trained model together in an
interactive Streamlit application. The dashboard uses a dark,
premium-style interface.

Dashboard pages: 1. **Business Overview** --- high-level review and
rating indicators. 2. **Customer Feedback Explorer** --- explore
customer feedback and review information. 3. **Issue Prioritization**
--- examine topic-level priorities and supporting evidence. 4.
**Sentiment Intelligence** --- review sentiment distributions and model
information. 5. **AI Prediction Studio** --- enter review text and
obtain a model sentiment prediction.

The dashboard loads the processed data, analysis summaries, and saved
machine-learning model from the project folders.

Run the dashboard locally:

``` bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

``` text
http://localhost:8501
```

### Phase 5 --- Final Evaluation, Documentation and Deployment

**Status: Planned**

Planned tasks: - Complete a focused evaluation of the selected model. -
Review negative and neutral class performance. - Document model
limitations and possible improvements. - Review the dashboard using the
final local dataset and model. - Finalise repository documentation and
dependencies. - Prepare the repository for public deployment. - Deploy
the Streamlit application. - Test the deployed application and add its
public URL to this README. - Add the project to the developer portfolio
and professional profiles.

------------------------------------------------------------------------

## Project Structure

``` text
ReviewRadar-AI/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   └── clothing_reviews.csv
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
│   ├── sentiment_classification.py
│   └── model_comparison.py
│
├── models/
│   ├── sentiment_model.pkl
│   └── best_sentiment_model_day5.pkl
│
├── reports/
│   ├── figures/
│   ├── sentiment_classification_report.csv
│   ├── sentiment_metrics.json
│   ├── model_comparison.csv
│   ├── model_classification_details.csv
│   └── best_model_metrics.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above shows the main project files and generated outputs.
Some generated files, including raw and processed CSV datasets, may be
excluded from GitHub through `.gitignore`.

------------------------------------------------------------------------

## Getting Started

### 1. Clone the repository

``` bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ReviewRadar-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual repository URL.

### 2. Create a virtual environment

Windows PowerShell:

``` powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

If Python 3.13 is not installed, use an installed Python version
compatible with the project's dependencies.

### 3. Install dependencies

``` powershell
python -m pip install -r requirements.txt
```

If a `requirements.txt` file is not yet available, install the core
packages:

``` powershell
python -m pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit
```

### 4. Add the dataset

Place the downloaded dataset at:

``` text
data/raw/clothing_reviews.csv
```

### 5. Run the dashboard

``` powershell
python -m streamlit run app.py
```

The dashboard requires the processed data files and saved model to be
present in their expected project paths. The dashboard command does not
automatically recreate these outputs.

------------------------------------------------------------------------

## Dashboard Pages

  -----------------------------------------------------------------------
  Page                                Purpose
  ----------------------------------- -----------------------------------
  Business Overview                   Summarises the customer-review
                                      dataset and key indicators.

  Customer Feedback Explorer          Supports exploration of customer
                                      reviews.

  Issue Prioritization                Presents topic-level issue
                                      priorities and review evidence.

  Sentiment Intelligence              Presents sentiment analysis and
                                      model evaluation information.

  AI Prediction Studio                Predicts sentiment for newly
                                      entered review text using the saved
                                      model.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Methodology and Important Notes

### Rating-derived sentiment labels

Sentiment classes were generated from star ratings rather than manually
annotated text: - 1--2 stars are treated as negative. - 3 stars are
treated as neutral. - 4--5 stars are treated as positive.

Therefore, model performance measures agreement with these
rating-derived labels. They should not be interpreted as proof of
agreement with independent human sentiment annotations.

### Topic modelling

NMF topics are discovered from patterns in TF-IDF features. Topic labels
were assigned by reviewing the topic keywords. These labels are
descriptive interpretations of the discovered word patterns, not
predefined ground-truth categories.

### Time-based analysis

The project does not claim to identify changes over time or trends
because the analysis described here does not use a verified review-date
field.

### Issue-prioritisation score

The priority score is a project-defined heuristic that combines review
volume, non-recommendation percentage, and rating dissatisfaction. It is
intended to help organise further investigation, not to establish
causation or a validated level of business risk.

------------------------------------------------------------------------

## Current Results

### Dataset overview

-   Total reviews: 23,486
-   Reviews with usable text: 22,641
-   Average rating: approximately 4.2
-   Recommended reviews: 19,314
-   Not-recommended reviews: 4,172

### Sentiment label distribution

  Sentiment     Number of reviews
  ----------- -------------------
  Positive                 17,448
  Neutral                   2,823
  Negative                  2,370
  **Total**            **22,641**

### Baseline Logistic Regression results

  Metric                Result
  ------------------- --------
  Accuracy              76.97%
  Macro precision         0.58
  Macro recall            0.64
  Macro F1-score          0.60
  Weighted F1-score       0.79

The baseline model performed best on the positive class. Negative and
neutral classes were more challenging.

### Model comparison

Three models were compared: - Logistic Regression - Linear SVC -
Complement Naive Bayes

**Complement Naive Bayes was selected based on macro F1-score.** Exact
comparison scores should be taken from `reports/model_comparison.csv`;
they are not reproduced here.

------------------------------------------------------------------------

## Limitations and Future Work

-   Sentiment labels are derived from star ratings, not independent
    manual annotations.
-   The dataset is imbalanced, with substantially more positive reviews
    than neutral or negative reviews.
-   Topic labels are assigned through interpretation of NMF keywords.
-   The issue-prioritisation formula is a heuristic and has not been
    validated against business outcomes.
-   Findings reflect this dataset and should not automatically be
    generalised to all e-commerce businesses.
-   Future work may include stronger evaluation, improved handling of
    minority sentiment classes, additional datasets, and deployment
    monitoring.

------------------------------------------------------------------------

## Author

**Hamna Fazil**

ReviewRadar AI --- Intelligent Customer Feedback Analysis

GitHub: Add your repository profile or project URL here.

Portfolio: https://hamnafazil.github.io/HamnaFazil/

------------------------------------------------------------------------

*This project is developed for learning, portfolio development, and
demonstration of data analytics, natural-language processing, machine
learning, and business-intelligence skills.*
