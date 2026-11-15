# 💼 Enterprise Job Market Analysis & ML Recommendation Engine

> **Live Local Application:** [  https://project8-gbz97mrgp7vzfkzb6zyhij.streamlit.app/](http://localhost:8502)  
> **Author:** Govind Mishra

---

## 🚀 Overview

This repository hosts an interactive machine learning application for enterprise freelance job market analytics, salary forecasting, and candidate skill-job matching trained on **244,828 Upwork job postings**.

### Key Features:
1. 🤖 **Automated Salary Estimator (Pipeline):** Real-time hourly wage prediction using a serialized Scikit-Learn `ColumnTransformer` + $L_2$ Regularized Ridge Regression pipeline (`model_pipeline.joblib`).
2. 🎯 **Personalized Recommendation Engine:** High-speed candidate skill-job matching over 240,000+ job postings using an 8,000-dimensional TF-IDF vector space with Cosine Similarity ranking (`rec_engine.joblib`).

---

## 📂 Repository File Structure (Essential Files to Upload)

To run this application on **GitHub**, **Streamlit Community Cloud**, or any local/cloud server, the following files are required:

| File | Purpose | Size | Required? |
| :--- | :--- | :---: | :---: |
| **`app.py`** | Main Streamlit web application code | ~4 KB | **MANDATORY** |
| **`requirements.txt`** | Lightweight Python dependencies for Streamlit serving | ~100 B | **MANDATORY** |
| **`model_pipeline.joblib`** | Pre-trained Scikit-Learn salary estimation pipeline | ~126 KB | **MANDATORY** |
| **`rec_engine.joblib`** | Pre-computed TF-IDF matrix & candidate dataset for recommendations | ~7.9 MB | **MANDATORY** |
| **`start_app.bat`** | Windows one-click batch startup script | ~200 B | Recommended |
| **`start_app.ps1`** | PowerShell startup script | ~250 B | Recommended |
| **`Dockerfile`** | Containerization image specification | ~800 B | Recommended |
| **`docker-compose.yml`** | Multi-container production deployment | ~350 B | Recommended |
| **`project_8.ipynb`** | Full end-to-end data science Jupyter notebook (EDA, Deep Learning, ML) | ~3.5 MB | Project Artifact |
| **`Job_Market_Analytics_Executive_Presentation.pptx`** | 24-slide executive client presentation | ~2.4 MB | Project Artifact |


---

## ⚡ Quick Start / How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py --server.port https://project8-gbz97mrgp7vzfkzb6zyhij.streamlit.app/
```
Or simply double-click **`start_app.bat`** on Windows!

Open your browser at **`https://project8-gbz97mrgp7vzfkzb6zyhij.streamlit.app/`**.

