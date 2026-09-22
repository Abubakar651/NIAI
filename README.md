# NAVTTC / NIAI — AI & Data Analytics Coursework & Projects

Repository containing coursework, practical lab exercises, and mini-projects completed during the **NAVTTC / National Institute of Artificial Intelligence (NIAI)** program.

---

## 📂 Repository Structure

```text
├── SQL_mini_project-LMS_Students/     # Learning Management System (LMS) SQL & Python project
│   ├── lms_database.sql               # Database schema, table definitions, and queries
│   ├── db.py                          # Python MySQL database connector & CRUD operations
│   ├── requirements.txt               # Python package dependencies
│   └── SQL Mini Project-LMS-Students.pdf # Project requirements & specifications
│
├── alfatah_store/                     # Al-Fatah Store Retail Management System
│   ├── app.py                         # Application entrypoint / UI
│   ├── database.py                    # Database handler & logic
│   ├── data/                          # Orders & products data
│   └── README.md                      # Project documentation
│
├── Road_Accident_Mini_Project/        # Road Accident Analysis Mini-Project
│   ├── Road_Accident_Dataset.xlsx     # Raw dataset
│   ├── Road_Accident_Dataset(completed).xlsx # Processed / completed dataset
│   └── Road_Accident_Mini_Project_Requirements-Students.pdf
│
├── Day 3 Advanced SQL/                # Advanced SQL Lectures & Labs
│   ├── Advanced SQL/
│   │   ├── Advanced SQL Queries.sql   # Complex joins, window functions, and analytics queries
│   │   ├── Advanced SQL 1.pptx        # Presentation slides (Part 1)
│   │   └── Advanced SQL 2.pptx        # Presentation slides (Part 2)
│   └── mysql connector.ipynb          # Jupyter notebook connecting Python to MySQL
│
├── Day 4/                             # Data Transformation & ETL (Power Query)
│   ├── Power Query/                   # Multi-source data transformation exercises
│   └── Use Case-ITW/                  # International Telecom Week (ITW) use-case datasets
│
├── Datasets_and_Exercises/            # Machine Learning & Statistics Practice Datasets
│   ├── Dataset Spam No Spam Students.xlsx # Classification practice dataset (raw)
│   ├── Dataset Spam No Spam Students(completed).xlsx # Classification dataset (completed)
│   └── Statistics Exercise Students.xlsx   # Descriptive & inferential statistics exercises
│
├── Numpy.ipynb                        # Complete NumPy guide: arrays, vectorization, broadcasting & linear algebra
├── Pandas.ipynb                       # Comprehensive Pandas guide: DataFrames, ETL, cleaning & transformations
├── Matplotlib.ipynb                   # Data visualization guide: line, bar, scatter, subplots & exports
├── Seaborn.ipynb                      # Statistical data visualization: distributions, categorical & correlation heatmaps
├── Scikit_Learn.ipynb                 # Machine Learning pipeline: preprocessing, models, metrics & pipelines
├── sales_trend.png                    # Sample visualization export from Matplotlib notebook
├── coffee_machine_simulator.py        # OOP Coffee Machine simulator with coin & resource logic
├── Even.py                            # Basic Python practice (conditional logic)
├── area_of_circle.py                  # Basic Python practice (geometry calculations)
├── MASTER_SCHEDULE.md                 # Daily study plan, Jira targets, and timetable
└── .gitignore                         # Git exclusion rules for large media, venvs, and secrets
```

---

## 🛠️ Tech Stack & Skills

- **Languages**: Python, SQL
- **Databases**: MySQL, Relational Schema Design, Foreign Keys & Constraints
- **Data Analysis & BI**: Microsoft Power BI, Power Query, Microsoft Excel
- **Data Transformation**: ETL workflows, unpivoting, merging, and cleaning multi-source datasets
- **Data Science & ML Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- **Database & Web Connectors**: `mysql-connector-python`, `openpyxl`, `streamlit`

---

## 🚀 Key Mini-Projects

### 1. [LMS Database Management System](SQL_mini_project-LMS_Students/)
A relational database system modeling a Learning Management System (students, courses, enrollments, instructors, and grades), paired with a Python client for database connectivity.

### 2. [Al-Fatah Retail Store System](alfatah_store/)
A store management application managing inventory, product catalogs, customer transactions, and order logs.

### 3. [Road Accident Analysis](Road_Accident_Mini_Project/)
An end-to-end data preparation and reporting mini-project analyzing road accident causes, casualties, road conditions, and trends.

---

## ⚙️ Setup & Installation

To run the Python scripts or database connectors locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abubakar651/NIAI.git
   cd NIAI
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r SQL_mini_project-LMS_Students/requirements.txt
   ```
