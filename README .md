
#  Customer Churn Prediction — End-to-End Machine Learning Project

##  Business Problem
Customer churn is a major revenue loss for subscription-based businesses. Acquiring new customers is much more expensive than retaining existing ones.  
The goal of this project is to predict which customers are likely to churn so that the business can proactively intervene and retain them.

This is framed as a binary classification problem:
- 1 = Customer will churn
- 0 = Customer will stay

---

##  Dataset
The dataset contains customer-level information from a telecom company, including:
- Customer demographics (gender, senior citizen, partner, dependents)
- Services subscribed (internet, phone, streaming, security, etc.)
- Contract & billing information (contract type, tenure, payment method, charges)
- Target variable: Churn (Yes / No)

The dataset is imbalanced, making accuracy an inappropriate metric.


##  Solution Approach

The project is built as a **complete ML pipeline**:

1. **Data Understanding**
   - Inspected schema, data types, missing values
   - Identified data quality issue in `TotalCharges`

2. **Exploratory Data Analysis (EDA)**
   - Analyzed churn by contract type, tenure, charges, and services
   - Discovered that month-to-month contracts, short tenure, and high charges strongly increase churn

3. **Feature Engineering & Preprocessing**
   - Fixed data types and missing values
   - Created business-driven features such as:
     - charges_per_tenure
     - is_long_term_customer
   - One-hot encoded categorical variables and scaled numeric features

4. **Modeling & Evaluation**
   - Trained and compared:
     - Logistic Regression
     - Random Forest
     - XGBoost
   - Evaluated using:
     - ROC AUC
     - Recall for churn class
   - Selected models based on **business objective**, not accuracy

5. **Business Threshold Optimization (Key Part)**
   - Default threshold (0.5) is not optimal for business
   - Built a **cost-sensitive decision framework**:
     - Missing a churner = very high cost
     - Contacting a loyal customer = low cost
   - Optimized the classification threshold to **minimize total business loss**

---

##  Business Impact

By optimizing the decision threshold:

- Churn recall increased to **~94%**
- Missed churners reduced drastically
- Total simulated business loss was **reduced significantly** compared to default threshold

Final system:
> **Logistic Regression + Business-optimized threshold**

---

##  Project Structure

- notebooks/1 Data_overview.ipynb Data understanding & quality checks  
- notebooks/2 Eda.ipynb Business-focused EDA  
- notebooks/3 Feature_engineering and Pre-Processing.ipynb Feature engineering pipeline  
- notebooks/4 Baseline-Models.ipynb Model training & comparison  
- notebooks/5 Threshold_optimization.ipynb Cost-sensitive threshold optimization  

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

##  Final Note

This project focuses not only on predictive performance, but on **making ML decisions that directly optimize business outcomes**.
