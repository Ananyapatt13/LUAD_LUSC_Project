import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

print("Running from:", os.path.abspath(__file__))

# 1. Load Data
expr = pd.read_csv("../data/expression.csv", index_col=0).T
labels = pd.read_csv("../data/labels.csv")

# Align samples
expr = expr.loc[labels["sample_id"]]
y = labels["cancer_type"]

# 2. Log-Transformation
# Apply log2(x+1) transformation to normalize count distribution
print("Applying log-transformation...")
expr_log = np.log1p(expr)

# 3. Train-Test Split
# Split data prior to feature selection/scaling to prevent data leakage
X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
    expr_log,
    y,
    labels["sample_id"],
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Feature Selection
# Remove low-variance features based on training set statistics
print("Filtering features...")
selector = VarianceThreshold(threshold=0.1)
selector.fit(X_train_raw)

X_train_sel = selector.transform(X_train_raw)
X_test_sel = selector.transform(X_test_raw)

# 5. Standardization
# Z-score normalization fitted on training set
print("Scaling features...")
scaler = StandardScaler()
scaler.fit(X_train_sel)

X_train = scaler.transform(X_train_sel)
X_test = scaler.transform(X_test_sel)

# 6. Model Training
print("Training SVM...")
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {acc:.4f}")

# 8. Save Artifacts
joblib.dump(model, "../results/svm_model.pkl")
joblib.dump(selector, "../results/variance_selector.pkl")
joblib.dump(scaler, "../results/scaler.pkl")

# Save split indices for reproducibility
np.save("../results/train_indices.npy", np.array(train_idx))
np.save("../results/test_indices.npy", np.array(test_idx))

# 9. Visualization & Analysis
# Generate and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["LUAD", "LUSC"],
            yticklabels=["LUAD", "LUSC"])
plt.title("Confusion Matrix")
plt.savefig("../plots/confusion_matrix.png", dpi=300)
plt.close()

# Extract and save top predictive features
weights = model.coef_[0]
gene_names = expr.columns[selector.get_support()]

gene_weights = pd.DataFrame({"gene": gene_names, "weight": weights})
# Sort genes by absolute weight magnitude
gene_weights = gene_weights.iloc[
    gene_weights["weight"].abs().sort_values(ascending=False).index
]
gene_weights.to_csv("../results/top_genes.csv", index=False)

print("Pipeline execution complete. Results saved.")