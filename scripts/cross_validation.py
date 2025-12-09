import os
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

print(f"Running cross-validation from: {os.path.abspath(__file__)}")

# --- CONFIGURATION ---
DATA_DIR = "../data"

# 1. Load Data (Memory Optimized)
print("Loading data...")
# Read as float32 immediately to save memory
expr = pd.read_csv(os.path.join(DATA_DIR, "expression.csv"), index_col=0).T.astype(np.float32)
labels = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

# Align samples
expr = expr.loc[labels["sample_id"]]
y = labels["cancer_type"]

# 2. Log-Transformation & Numpy Conversion
print("Applying log-transformation and converting to numpy...")
# Convert to numpy array immediately to avoid Pandas overhead during CV
X = np.log1p(expr).values.astype(np.float32)
y = y.values

# Free up memory
del expr
gc.collect()

# 3. Define Pipeline
pipeline = Pipeline([
    ('selector', VarianceThreshold(threshold=0.1)),
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel="linear"))
])

# 4. Run 5-Fold Cross-Validation
print("Executing Stratified 5-Fold Cross-Validation (Sequential)...")

# n_jobs=1 prevents memory duplication (essential for large datasets on Windows)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    pipeline, 
    X, 
    y, 
    cv=cv, 
    scoring='accuracy', 
    n_jobs=1  # Changed from -1 to 1 to fix MemoryError
)

# 5. Report Results
print("\n------------------------------------------------")
print("Cross-Validation Results (Accuracy)")
print("------------------------------------------------")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.4f}")

print("------------------------------------------------")
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")
print("------------------------------------------------")