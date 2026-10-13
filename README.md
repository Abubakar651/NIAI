# NAVTTC / NIAI — AI, Data Science & Analytics Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-blue.svg)](https://www.mysql.com/)
[![Power BI](https://img.shields.io/badge/Power_BI-Analytics-yellow.svg)](https://powerbi.microsoft.com/)

Central repository containing end-to-end projects, coursework labs, and practical implementations completed during the **NAVTTC / National Institute of Artificial Intelligence (NIAI)** program hosted at **NETSOL Technologies**.

---

## 📂 Repository Architecture

```text
NIAI/
├── 📁 NAVTCC exercise/                 # 🚀 Official NAVTTC Weekly Hands-on Tasks (Weeks 1–5)
│   ├── 📁 Week1/                       # Python fundamentals, Linux, strings, loops, collections
│   ├── 📁 Week2/                       # Functions, exceptions, custom errors, descriptive stats
│   ├── 📁 Week3/                       # NumPy array computing, indexing, slicing, reshaping
│   ├── 📁 Week4/                       # Pandas DataFrames, CSV analysis, data cleaning & correlation
│   └── 📁 Week5/                       # Scikit-Learn Linear Regression & Logistic Regression
│
├── 📁 Projects/                        # ⭐ Production-grade showcase projects
│   ├── 📁 Credit_Score_EDA/            # Credit risk exploratory data analysis & quality audit
│   ├── 📁 AlFatah_Retail_Store/        # Retail Supermarket management system & order processing
│   ├── 📁 LMS_Database_System/         # Relational LMS database schema (MySQL) & Python CRUD
│   └── 📁 Road_Accident_Analytics/     # Incident causality & casualty reporting workbook
│
├── 📁 Coursework/                      # 📚 Modular curriculum labs aligned with syllabus
│   ├── 📁 Python_and_OOP/              # OOP principles, Data Structures & Algorithms, CLI tools
│   ├── 📁 SQL_and_Databases/           # Advanced SQL queries, window functions & MySQL connector
│   ├── 📁 Data_Science_Toolkit/        # NumPy, Pandas, Matplotlib & Seaborn deep-dive notebooks
│   ├── 📁 Power_Query_and_BI/          # Multi-source ETL pipelines & relational data modeling
│   └── 📁 Machine_Learning/            # Supervised classification, pipelines & statistical exercises
│
├── 📁 Resources/                       # 📖 Curricula & reference guides
│   ├── NAVTTC Outline-Students.pdf     # Official 12-week AI & Data Analytics syllabus
│   └── MASTER_SCHEDULE.md              # Daily 9-hour study plan & timetable
│
├── .gitignore                          # Production exclusions (large media, venvs, secrets)
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## 🚀 NAVTTC Weekly Practical Exercises (Weeks 1 to 5)

Organized and standalone practice scripts following the official [NAVTTC Outline](Resources/NAVTTC%20Outline-Students.pdf):

| Week | Directory | Focus & Core Tasks |
| :--- | :--- | :--- |
| **Week 1** | [`NAVTCC exercise/Week1/`](NAVTCC%20exercise/Week1/) | **Python Basics & Linux**: Installation guide, terminal commands, data types, string operations, conditionals, loops (`for`, `while`, `break`, `continue`, `enumerate`), lists, tuples, sets, dictionaries (Tasks 1–20) |
| **Week 2** | [`NAVTCC exercise/Week2/`](NAVTCC%20exercise/Week2/) | **Functions, Exceptions & Stats**: User-defined functions, error handling (`try/except/finally`), custom user exceptions, central tendency (mean, median, mode) & dispersion (std dev, variance) (Tasks 21–23, 49, 50) |
| **Week 3** | [`NAVTCC exercise/Week3/`](NAVTCC%20exercise/Week3/) | **NumPy Array Computing**: Creating arrays (0D–3D), indexing, slicing, data types, copy vs view, shape, reshape, iterating (`nditer`), joining (`hstack`/`vstack`), splitting, searching (`where`), sorting, random sampling (Tasks 28–40) |
| **Week 4** | [`NAVTCC exercise/Week4/`](NAVTCC%20exercise/Week4/) | **Pandas Data Analysis**: Series, DataFrames, `.loc` selection, CSV ingestion, data profiling (`describe`, `info`), missing value handling (`fillna`), deduplication, Pearson correlation matrices (Tasks 41–48) |
| **Week 5** | [`NAVTCC exercise/Week5/`](NAVTCC%20exercise/Week5/) | **Classical Machine Learning**: Scikit-Learn `LinearRegression` (train/test split, slope, intercept, MAE, MSE, $R^2$) and `LogisticRegression` (binary classification, confusion matrix, classification report) (Tasks 51–52) |

---

## ⭐ Featured Projects

### 1. [Credit Score Risk & EDA Assessment](Projects/Credit_Score_EDA/)
* **Focus**: Data Quality Assessment, Credit Bureau Delinquency, Outlier Winsorization & ML Preparation.
* **Dataset**: 150,000 borrower records across 13 financial and behavioral metrics (`CreditScore.csv`).
* **Deliverables**: Complete 39-cell notebook ([`Credit_score_EDA.ipynb`](Projects/Credit_Score_EDA/Credit_score_EDA.ipynb)) with pre-rendered charts, structured Data Quality Report, univariate/bivariate visual EDA with key insights per plot, correlation matrix, clean data pipeline, and pre-modeling ML checklist.
* **Stack**: Python, `pandas`, `numpy`, `matplotlib`, `seaborn`.

### 2. [Al-Fatah Retail Store System](Projects/AlFatah_Retail_Store/)
* **Focus**: Supermarket Inventory Tracking, POS Billing, 10% Discount Rule & Sales Dashboard.
* **Features**: Dual interface with both a standalone terminal console app ([`main.py`](Projects/AlFatah_Retail_Store/main.py)) and a full Streamlit web app ([`app.py`](Projects/AlFatah_Retail_Store/app.py)), atomic stock validation, automatic 10% discount on purchases $\ge$ Rs. 1,000, itemized receipt printing, full Admin CRUD, and sales/inventory visual charts.
* **Stack**: Python, Streamlit, Pandas, CSV persistence.

### 3. [LMS Database Management System](Projects/LMS_Database_System/)
* **Focus**: Relational Database Design, Normalization, Constraints & Streamlit Web App.
* **Features**: Complete 5-table MySQL relational schema ([`lms_database.sql`](Projects/LMS_Database_System/lms_database.sql)) with primary/foreign keys and duplicate enrollment prevention, parameterized Python CRUD backend ([`db.py`](Projects/LMS_Database_System/db.py)), full Streamlit frontend ([`app.py`](Projects/LMS_Database_System/app.py)), and SQL reports answering all 6 core business questions (BR-01 through BR-06).
* **Stack**: MySQL, SQL DDL/DML, `mysql-connector-python`, Python, Streamlit, Pandas.

### 4. [Road Accident Analytics](Projects/Road_Accident_Analytics/)
* **Focus**: Public Safety Incident Analysis, Casualty Segmentation, Environmental Correlation.
* **Features**: Multi-variable analysis correlating weather conditions, road surface types, vehicle classifications, and severity levels.
* **Stack**: Microsoft Excel, Power Query, Data Wrangling.

---

## 📚 Curriculum Coursework & Labs

| Module | Location | Core Topics Covered |
| :--- | :--- | :--- |
| **Python & OOP** | [`Coursework/Python_and_OOP/`](Coursework/Python_and_OOP/) | Core fundamentals (string manipulation, two-pointer algorithms, match-case calculator), MRO & diamond inheritance, property encapsulation, state machines (Coffee Maker), Student System, and internship assessments |
| **SQL & Databases** | [`Coursework/SQL_and_Databases/`](Coursework/SQL_and_Databases/) | CTEs, window functions, inner/outer joins, Python-to-MySQL pipeline |
| **Data Science Toolkit** | [`Coursework/Data_Science_Toolkit/`](Coursework/Data_Science_Toolkit/) | NumPy vectorized matrices, Pandas ETL transformations, Matplotlib figures, Seaborn statistical heatmaps |
| **Power Query & BI** | [`Coursework/Power_Query_and_BI/`](Coursework/Power_Query_and_BI/) | Relational modeling (`Exercise-1.xlsx`), unpivoting subcategorized tables, multi-country telecom consolidation |
| **Machine Learning** | [`Coursework/Machine_Learning/`](Coursework/Machine_Learning/) | Scikit-Learn workflows, Decision Trees (Shannon Entropy & Diabetes Diagnostic), WDBC diagnostic classification pipeline, Feature Engineering (KC House Data), Linear Regression (SAT/GPA, Salary/Exp), SMS spam classification, inferential statistics & probability distributions |

---

## ⚙️ Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abubakar651/NIAI.git
   cd NIAI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   # .venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📬 Contact & Feedback
For any queries, collaboration, or feedback regarding these exercises and projects:
- **Author**: Abubakar Iqbal
- **Email**: [abubakariqbal651@gmail.com](mailto:abubakariqbal651@gmail.com)
- **GitHub**: [@Abubakar651](https://github.com/Abubakar651)

---

## 📜 Curriculum Reference
The full 12-week course outline covering Linux, Python, Machine Learning, Deep Learning (CNN/RNN/LSTM), and Microsoft Azure AI is documented in [`Resources/NAVTTC Outline-Students.pdf`](Resources/NAVTTC%20Outline-Students.pdf).
