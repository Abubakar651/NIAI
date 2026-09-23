# NAVTTC / NIAI — AI, Data Science & Analytics Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-blue.svg)](https://www.mysql.com/)
[![Power BI](https://img.shields.io/badge/Power_BI-Analytics-yellow.svg)](https://powerbi.microsoft.com/)

Central repository containing end-to-end projects, coursework labs, and practical implementations completed during the **NAVTTC / National Institute of Artificial Intelligence (NIAI)** program.

---

## 📂 Repository Architecture

```text
NIAI/
├── 📁 Projects/                         # ⭐ Production-grade showcase projects
│   ├── 📁 Credit_Score_EDA/             # Credit risk exploratory data analysis & quality audit
│   ├── 📁 AlFatah_Retail_Store/         # Retail Supermarket management system & order processing
│   ├── 📁 LMS_Database_System/          # Relational LMS database schema (MySQL) & Python CRUD
│   └── 📁 Road_Accident_Analytics/      # Incident causality & casualty reporting workbook
│
├── 📁 Coursework/                       # 📚 Modular curriculum labs aligned with syllabus
│   ├── 📁 Python_and_OOP/               # OOP principles, Data Structures & Algorithms, CLI tools
│   ├── 📁 SQL_and_Databases/            # Advanced SQL queries, window functions & MySQL connector
│   ├── 📁 Data_Science_Toolkit/         # NumPy, Pandas, Matplotlib & Seaborn deep-dive notebooks
│   ├── 📁 Power_Query_and_BI/           # Multi-source ETL pipelines & relational data modeling
│   └── 📁 Machine_Learning/             # Supervised classification, pipelines & statistical exercises
│
├── 📁 Resources/                        # 📖 Curricula & reference guides
│   ├── NAVTTC Outline-Students.pdf      # Official 12-week AI & Data Analytics syllabus
│   └── MASTER_SCHEDULE.md               # Daily 9-hour study plan & timetable
│
├── .gitignore                           # Production exclusions (large media, venvs, secrets)
├── requirements.txt                     # Project dependencies
└── README.md
```

---

## ⭐ Featured Projects

### 1. [Credit Score Risk & EDA Assessment](Projects/Credit_Score_EDA/)
* **Focus**: Data Quality Assessment, Credit Bureau Delinquency, Outlier & Anomaly Detection.
* **Dataset**: 150,000 borrower records across 13 financial and behavioral metrics.
* **Key Findings**: Severe class imbalance (6.68% default rate), non-linear relationship between age and risk (young borrowers 25–29 represent peak default risk at 12.1%), identification of special encoding flags (`98` past-due codes).
* **Stack**: `pandas`, `numpy`, `matplotlib`, `seaborn`.

### 2. [Al-Fatah Retail Store System](Projects/AlFatah_Retail_Store/)
* **Focus**: Object-Oriented Software Design, Inventory Tracking & Transaction Management.
* **Features**: Dynamic cart calculations, automated tier-based discounts, CSV-backed persistence, and order management.
* **Stack**: Python, OOP, CSV data handling.

### 3. [LMS Database Management System](Projects/LMS_Database_System/)
* **Focus**: Relational Database Design, Normalization, Integrity Constraints & Python Integration.
* **Features**: Complete SQL schema modeling courses, students, enrollments, instructors, and grades with Python MySQL connector integration.
* **Stack**: MySQL, SQL DDL/DML, `mysql-connector-python`.

### 4. [Road Accident Analytics](Projects/Road_Accident_Analytics/)
* **Focus**: Public Safety Incident Analysis, Casualty Segmentation, Environmental Correlation.
* **Features**: Multi-variable analysis correlating weather conditions, road surface types, vehicle classifications, and severity levels.
* **Stack**: Microsoft Excel, Power Query, Data Wrangling.

---

## 📚 Curriculum Coursework & Labs

| Module | Location | Core Topics Covered |
| :--- | :--- | :--- |
| **Python & OOP** | [`Coursework/Python_and_OOP/`](Coursework/Python_and_OOP/) | Classes, inheritance, state machines (Coffee Maker), Student System, Binary Search Trees & Big-O notation |
| **SQL & Databases** | [`Coursework/SQL_and_Databases/`](Coursework/SQL_and_Databases/) | CTEs, window functions, inner/outer joins, Python-to-MySQL pipeline |
| **Data Science Toolkit** | [`Coursework/Data_Science_Toolkit/`](Coursework/Data_Science_Toolkit/) | NumPy vectorized matrices, Pandas ETL transformations, Matplotlib figures, Seaborn statistical heatmaps |
| **Power Query & BI** | [`Coursework/Power_Query_and_BI/`](Coursework/Power_Query_and_BI/) | Relational modeling (`Exercise-1.xlsx`), unpivoting subcategorized tables, multi-country telecom consolidation |
| **Machine Learning** | [`Coursework/Machine_Learning/`](Coursework/Machine_Learning/) | Scikit-Learn workflows, SMS spam classification, inferential statistics & probability distributions |

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

## 📜 Curriculum Reference
The full 12-week course outline covering Linux, Python, Machine Learning, Deep Learning (CNN/RNN/LSTM), and Microsoft Azure AI is documented in [`Resources/NAVTTC Outline-Students.pdf`](Resources/NAVTTC%20Outline-Students.pdf).
