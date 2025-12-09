import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA

# Configure paths
DATA_DIR = "../data"
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots"

print(f"Running evaluation from: {os.path.abspath(__file__)}")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Load Data
if not os.path.exists(os.path.join(DATA_DIR, "expression.csv")):
    raise FileNotFoundError("expression.csv not found.")

expr = pd.read_csv(os.path.join(DATA_DIR, "expression.csv"), index_col=0).T
labels = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

sample_ids = labels["sample_id"]
expr = expr.loc[sample_ids]
y = labels["cancer_type"]
y.index = sample_ids

# 2. Preprocessing
print("Applying log-transformation...")
expr_log = np.log1p(expr)

# 3. Load Artifacts
print("Loading training artifacts...")
try:
    model = joblib.load(os.path.join(RESULTS_DIR, "svm_model.pkl"))
    selector = joblib.load(os.path.join(RESULTS_DIR, "variance_selector.pkl"))
    scaler = joblib.load(os.path.join(RESULTS_DIR, "scaler.pkl"))
    test_idx = np.load(os.path.join(RESULTS_DIR, "test_indices.npy"), allow_pickle=True)
except FileNotFoundError:
    raise FileNotFoundError("Artifacts not found. Verify training is complete.")

test_idx = test_idx.astype(str)

# 4. Prepare Test Data
print("Transforming test set...")
X_test_raw = expr_log.loc[test_idx]
X_test_sel = selector.transform(X_test_raw)
X_test_scaled = scaler.transform(X_test_sel)

y_test = y.loc[test_idx]
y_test_bin = (y_test == "LUSC").astype(int)

# 5. Generate Visualizations

# --- PLOT 1: Confusion Matrix ---
print("Generating Confusion Matrix...")
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    cbar=False,
    square=True, # Confusion matrix must be square
    xticklabels=["LUAD", "LUSC"],
    yticklabels=["LUAD", "LUSC"],
    annot_kws={"size": 14, "weight": "bold"}
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# --- PLOT 2: PCA Plot (Square Geometry) ---
print("Generating PCA plot...")
pca = PCA(n_components=2)
coords = pca.fit_transform(X_test_scaled)
pca_df = pd.DataFrame({
    "PC1": coords[:, 0], 
    "PC2": coords[:, 1], 
    "Subtype": y_test.values
})

plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=pca_df, 
    x="PC1", 
    y="PC2", 
    hue="Subtype", 
    marker='o',       # Circles
    s=80,             
    alpha=0.7,        
    edgecolor='k',    
    linewidth=0.5
)
plt.title("PCA: Test Set Distribution")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('equal') # Forces true square geometry
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 3: ROC Curve ---
print("Generating ROC curve...")
proba_test = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test_bin, proba_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", lw=2.5, color='darkorange')
plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc.png"), dpi=300)
plt.close()

# --- PLOT 4: Feature Heatmap (Rectangular/Readable) ---
print("Generating feature heatmap...")
top_genes_path = os.path.join(RESULTS_DIR, "top_genes.csv")

if os.path.exists(top_genes_path):
    top_genes_df = pd.read_csv(top_genes_path)
    top_gene_names = top_genes_df.head(30)["gene"].values

    expr_top = expr_log[top_gene_names]
    expr_top_z = (expr_top - expr_top.mean()) / expr_top.std()

    order = np.argsort(y.values)
    expr_sorted = expr_top_z.iloc[order]

    # Rectangular size to fit genes readable
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        expr_sorted.T, 
        cmap="coolwarm", 
        center=0, 
        # square=False (Default) allows stretching
        xticklabels=False, 
        yticklabels=True,
        cbar_kws={"shrink": 0.5, "label": "Z-Score"}
    )

    plt.title("Top 30 Discriminatory Genes")
    plt.ylabel("Gene ID")
    plt.xlabel("Samples (Sorted by Subtype)")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("Top genes file not found. Skipping heatmap generation.")

print(f"Evaluation complete. Plots saved to {os.path.abspath(PLOTS_DIR)}")