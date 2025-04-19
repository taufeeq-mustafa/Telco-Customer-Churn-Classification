
# 📊 Telco Customer Churn Classification

This project aims to predict customer churn using a Telco dataset. The goal is to build classification models that accurately identify customers likely to cancel their service. The pipeline includes data preprocessing, handling class imbalance, model training, hyperparameter tuning, and model explainability using SHAP.

---

## 📁 Project Structure

```bash
├── Telco_Churn_Classification.ipynb   # Main code notebook
├── README.md                          # Project documentation
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset file (from Google Drive)
```

---

## 📌 Dataset

- **Source:** IBM Sample Dataset for Telco Customer Churn
- **Location:** `/content/drive/MyDrive/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target Column:** `Churn` (Yes/No)
- **Size:** ~7000 records, multiple features including demographics, account info, and service usage

---

## 🚀 Getting Started

### ✅ Prerequisites

Install the required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

Or use a Google Colab environment where most of these packages are pre-installed.

---

## 🧠 Workflow Summary

### 1. **Mount Google Drive & Load Data**

The dataset is mounted and loaded from Google Drive using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 2. **Exploratory Data Analysis**

- View data types and null values
- Distribution of the target variable (`Churn`)
- Visualizations using `Seaborn`

---

### 3. **Data Cleaning**

- Convert `TotalCharges` to numeric and handle missing values
- Encode the target: `Yes → 1`, `No → 0`

---

### 4. **Feature Engineering**

- Standard scaling of numeric features
- One-hot encoding of categorical features: `Contract`, `PaymentMethod`, `gender`

---

### 5. **Train-Test Split**

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

### 6. **Class Imbalance Handling**

Use **SMOTE** to balance the classes in the training set:

```python
from imblearn.over_sampling import SMOTE
```

---

### 7. **Model Training and Evaluation**

Models used:

- Logistic Regression
- Random Forest
- XGBoost

Metrics:

- Classification Report
- ROC-AUC Score

Example output:

```
----- Logistic Regression -----
Accuracy: 73%
ROC-AUC: 0.75
```

---

### 8. **Hyperparameter Tuning**

Performed for XGBoost using `GridSearchCV` with parameters:

```python
params = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "scale_pos_weight": [1, 5]
}
```

---

### 9. **Model Explainability (SHAP)**

Used **SHAP** to explain the predictions from the best model (XGBoost):

```python
shap.summary_plot(shap_values, X_test_processed[:100], ...)
```

Visualizations help interpret the impact of each feature on predictions.

---

## 📈 Results Summary

| Model              | Accuracy | ROC-AUC | Notes                         |
|-------------------|----------|---------|-------------------------------|
| Logistic Regression | 73%     | 0.75    | Best balance of recall/precision |
| Random Forest       | 73%     | 0.68    | Lower recall for churn class |
| XGBoost             | 75%     | 0.73    | Best overall performer        |

---

## 📌 Key Insights

- **XGBoost** performed best overall and was selected for further analysis.
- **SHAP** was useful in understanding the influence of features like `Contract` and `TotalCharges`.

---

## 📚 References

- [IBM Telco Customer Churn Dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

## ✨ Future Work

- Add more customer behavioral features.
- Deploy as a web app for real-time prediction.
- Integrate with business dashboards for live analytics.

---

## 🧑‍💻 Author

**Syed Taufeeq Mustafa**  
Final Year Software Engineering Student at SSUET  
Passionate about AI/ML and full-stack development.
