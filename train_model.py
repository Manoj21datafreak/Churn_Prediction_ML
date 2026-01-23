import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ----------------------------
# 1. Load Data
# ----------------------------
DATA_PATH = "Telco-Customer-Churn.csv"  # put csv in same folder

df = pd.read_csv(DATA_PATH)

# ----------------------------
# 2. Basic Cleaning
# ----------------------------

# Fix TotalCharges type issue
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop ID column
df = df.drop(columns=["customerID"])

# Encode target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ----------------------------
# 3. Feature Engineering
# ----------------------------

df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
df["is_long_term_customer"] = (df["tenure"] > 24).astype(int)

# ----------------------------
# 4. Split X and y
# ----------------------------

y = df["Churn"]
X = df.drop(columns=["Churn"])

# Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# ----------------------------
# 5. Preprocessing Pipelines
# ----------------------------

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ----------------------------
# 6. Model
# ----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# ----------------------------
# 7. Full Pipeline
# ----------------------------

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# ----------------------------
# 8. Train/Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 9. Train
# ----------------------------

print("Training model...")
pipeline.fit(X_train, y_train)

# ----------------------------
# 10. Quick Evaluation
# ----------------------------

y_proba = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("Validation ROC AUC:", auc)

# ----------------------------
# 11. Save Model
# ----------------------------

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/churn_model.joblib")

print("Model saved to model/churn_model.joblib")
