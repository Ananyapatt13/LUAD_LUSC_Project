import os
import textwrap

import numpy as np
import pandas as pd
import gseapy as gp
import mygene
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
DATA_DIR = "../data"
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots"
GO_DB = "GO_Biological_Process_2021"
N_GENES = 150   # number of upregulated genes per subtype to use

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Starting LUAD/LUSC GO enrichment (DE-based)...")

# ----------------- HELPERS -----------------
def wrap_labels(labels, width=40):
    return ["\n".join(textwrap.wrap(str(l), width)) for l in labels]


def convert_ids_to_symbols(ensembl_ids):
    """Convert Ensembl gene IDs to gene symbols using MyGeneInfo."""
    base_ids = [g.split(".")[0] for g in ensembl_ids]  # drop version suffix
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        base_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        verbose=False,
    )
    return [r["symbol"] for r in res if "symbol" in r]


def run_enrich_and_plot(symbols, cancer_type, palette):
    """Run Enrichr GO BP and save CSV + barplot."""
    print(f"\nProcessing {cancer_type} ({len(symbols)} genes)...")

    enr = gp.enrichr(
        gene_list=symbols,
        gene_sets=GO_DB,
        organism="human",
        outdir=None,
    )

    df = enr.res2d.copy()

    # Clean columns: keep main stats
    drop_cols = [
        "Old P-value",
        "Old Adjusted P-value",
        "Odds Ratio",
        "Combined Score",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Make Overlap Excel-safe (avoid date parsing)
    if "Overlap" in df.columns:
        df["Overlap"] = df["Overlap"].apply(lambda x: f" {x}")

    # Save full table
    csv_path = os.path.join(RESULTS_DIR, f"enrichment_{cancer_type}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV → {csv_path}")

    # ---- Barplot: top 5 by adjusted P-value ----
    df_plot = df.sort_values("Adjusted P-value").head(5).copy()
    df_plot["Significance"] = -np.log10(df_plot["Adjusted P-value"])
    df_plot["Term_wrapped"] = wrap_labels(df_plot["Term"], width=40)

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    sns.barplot(
        data=df_plot,
        x="Significance",
        y="Term_wrapped",
        palette=palette,
    )

    plt.title(f"{cancer_type}: Top Enriched Biological Processes",
              fontsize=16, fontweight="bold")
    plt.xlabel("-Log10(Adjusted P-value)", fontsize=13)
    plt.ylabel("")

    # If you want the FDR=0.05 line, uncomment:
    # plt.axvline(x=1.3, linestyle="--", color="black", alpha=0.6)

    plot_path = os.path.join(PLOTS_DIR, f"enrichment_{cancer_type}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved plot → {plot_path}")


# ----------------- MAIN LOGIC -----------------
# 1. Load expression (TPM) and labels
expr = pd.read_csv(os.path.join(DATA_DIR, "expression.csv"), index_col=0).T
labels = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

# Align samples
expr = expr.loc[labels["sample_id"]]

# 2. Log-transform TPM inside this script
expr_log = np.log1p(expr)

# Split LUAD/LUSC
luad_samples = labels.loc[labels["cancer_type"] == "LUAD", "sample_id"]
lusc_samples = labels.loc[labels["cancer_type"] == "LUSC", "sample_id"]

luad_mean = expr_log.loc[luad_samples].mean(axis=0)
lusc_mean = expr_log.loc[lusc_samples].mean(axis=0)

# 3. Differential expression: LUAD - LUSC
logFC = luad_mean - lusc_mean  # >0: LUAD-up, <0: LUSC-up

# 4. Get top upregulated genes for each subtype
luad_up = logFC.sort_values(ascending=False).head(N_GENES).index
lusc_up = logFC.sort_values(ascending=True).head(N_GENES).index

# 5. Convert to symbols + run enrichment
luad_symbols = convert_ids_to_symbols(luad_up)
lusc_symbols = convert_ids_to_symbols(lusc_up)

run_enrich_and_plot(lusc_symbols, "LUSC", "Reds_r")
run_enrich_and_plot(luad_symbols, "LUAD", "Blues_r")

print("\nGO enrichment (DE-based) finished.")
