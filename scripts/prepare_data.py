import os
import pandas as pd

print("Running from:", os.path.abspath(__file__))
print("Current dir:", os.getcwd())

# Load LUAD TPM file
luad_expr = pd.read_csv(
    "../data/TCGA-LUAD.star_tpm.tsv",
    sep="\t",
    index_col=0
)

print("Loaded LUAD TPM successfully!")
print("LUAD Shape:", luad_expr.shape)

# Load LUAD clinical file
luad_clin = pd.read_csv(
    "../data/TCGA-LUAD.clinical.tsv",
    sep="\t"
)

print("Loaded LUAD clinical successfully!")
print("LUAD Clinical Columns:", luad_clin.columns[:10])

# ---------------------------------------------------
# Load LUSC TPM file
# ---------------------------------------------------
lusc_expr = pd.read_csv(
    "../data/TCGA-LUSC.star_tpm.tsv",
    sep="\t",
    index_col=0
)

print("Loaded LUSC TPM successfully!")
print("LUSC Shape:", lusc_expr.shape)

# Load LUSC clinical file
lusc_clin = pd.read_csv(
    "../data/TCGA-LUSC.clinical.tsv",
    sep="\t"
)

print("Loaded LUSC clinical successfully!")
print("LUSC Clinical Columns:", lusc_clin.columns[:10])

# Combine expression matrices
expr_combined = pd.concat([luad_expr, lusc_expr], axis=1)

print("Combined shape:", expr_combined.shape)

# Save combined expression matrix
expr_combined.to_csv("../data/expression.csv")
print("Saved combined expression matrix as expression.csv")

# Build labels list
labels = []

# LUAD labels
for sample in luad_expr.columns:
    labels.append([sample, "LUAD"])

# LUSC labels
for sample in lusc_expr.columns:
    labels.append([sample, "LUSC"])

labels_df = pd.DataFrame(labels, columns=["sample_id", "cancer_type"])

labels_df.to_csv("../data/labels.csv", index=False)
print("Saved labels as labels.csv")
