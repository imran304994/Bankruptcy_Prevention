# Bankruptcy Prevention Prediction Platform

An end-to-end Machine Learning pipeline and web application designed to predict the likelihood of corporate bankruptcy based on critical business risk indicators. This project compares multiple supervised classification models and deploys the optimal model to provide real-time risk assessment.

## 📊 Business Risk Parameters Analyzed

The predictive models evaluate organizations across 6 major qualitative and quantitative risk dimensions:
* **Industrial Risk:** External factors affecting the specific industry sector.
* **Management Risk:** Operational stability and leadership efficiency.
* **Financial Flexibility:** The firm's capability to raise capital and manage debt burdens.
* **Credibility:** Trustworthiness, credit history, and market reputation.
* **Competitiveness:** Market share, technological edge, and positioning relative to rivals.
* **Operating Risk:** Day-to-day business operational hazards and asset utilization.

The target variable (`class`) classifies firms as **Bankruptcy** or **Non-Bankruptcy**.

## 🚀 Key Features

* **Statistical Feature Analysis:** Incorporates Chi-Square tests (`chi2_contingency`) to mathematically validate relationships between categorical risk factors.
* **Comparative Model Evaluation:** Evaluates a diverse suite of classifiers including Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM).
* **Hyperparameter Optimization:** Leverages `GridSearchCV` to automatically tune model parameters for optimal precision, recall, and ROC-AUC curves.
* **Interactive Deployment UI:** Implements a production-grade interactive dashboard using **Streamlit** for instant risk evaluation.
* **Serialized Model Pipe:** Exports trained mathematical weights securely via Python `pickle` serialization protocols for scalable local or cloud hosting.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Prep & Analysis:** Pandas, NumPy, Scikit-Learn (StandardScaler)
* **Statistical Modeling:** SciPy (Stats module)
* **Machine Learning Classifiers:** Logistic Regression, Decision Trees, Random Forests, SVM
* **Model Tuning & Serialization:** GridSearchCV, Pickle
* **Frontend UI App:** Streamlit
* **Data Visualization:** Matplotlib, Seaborn

## 📁 Project Structure

```text
├── data/
│   └── bankruptcy-prevention.csv        # Raw financial/risk indicators dataset
├── Bankruptcy_Prevention.ipynb          # End-to-end EDA and ML model training pipeline
├── Model.py                             # Streamlit web application dashboard code
├── bankruptcy_model.pkl                 # Trained production machine learning model
└── requirements.txt                     # System dependency library requirements
