import os
import pandas as pd
import numpy as np
import gc # Garbage Collector interface
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print(f"Running model comparison from: {os.path.abspath(__file__)}")

# 1. Load Data with Precision (float32)
print("Loading data (as float32)...")
expr = pd.read_csv("../data/expression.csv", index_col=0).T.astype(np.float32)
labels = pd.read_csv("../data/labels.csv")

# Align samples
expr = expr.loc[labels["sample_id"]]
y = labels["cancer_type"]

# 2. Log-Transformation
print("Applying log-transformation...")
expr = np.log1p(expr)

# 3. Train-Test Split
print("Splitting data...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    expr, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

del expr
gc.collect()

# 4. Feature Selection
print("Filtering low variance features...")
selector = VarianceThreshold(threshold=0.1)
selector.fit(X_train_raw)

X_train_sel = selector.transform(X_train_raw)
X_test_sel = selector.transform(X_test_raw)
del X_train_raw, X_test_raw
gc.collect()

# 5. Standardization
print("Standardizing features...")
scaler = StandardScaler()
scaler.fit(X_train_sel)

X_train = scaler.transform(X_train_sel)
X_test = scaler.transform(X_test_sel)

# Free up memory: delete unscaled selection
del X_train_sel, X_test_sel
gc.collect()

# 6. Define Models
models = {
    "SVM (Linear)": SVC(kernel="linear"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

print("\nModel Performance Benchmark (Test Set Accuracy)")

# 7. Evaluate
for name, clf in models.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   -> Accuracy: {acc:.4f}")

print("\nComparison complete.")