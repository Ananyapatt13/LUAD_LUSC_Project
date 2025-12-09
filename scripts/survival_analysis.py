import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ---------------- CONFIG ----------------
DATA_DIR = "../data"
PLOTS_DIR = "../plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Running survival analysis from: {os.path.abspath(__file__)}")

luad_path = os.path.join(DATA_DIR, "TCGA-LUAD.clinical.tsv")
lusc_path = os.path.join(DATA_DIR, "TCGA-LUSC.clinical.tsv")

if not (os.path.exists(luad_path) and os.path.exists(lusc_path)):
    raise FileNotFoundError("Clinical data files not found in ../data/")

# ---------------- LOAD CLINICAL ----------------
print("Loading clinical datasets...")
luad_df = pd.read_csv(luad_path, sep="\t")
lusc_df = pd.read_csv(lusc_path, sep="\t")

luad_df["cancer_type"] = "LUAD"
lusc_df["cancer_type"] = "LUSC"

clinical = pd.concat([luad_df, lusc_df], ignore_index=True)

# If case_id exists, drop duplicates to avoid double-counting
for id_col in ["case_id", "submitter_id", "id"]:
    if id_col in clinical.columns:
        clinical = clinical.drop_duplicates(subset=id_col)
        break

# ---------------- SURVIVAL FIELDS ----------------
print("Processing survival metrics...")

# 1. Vital status
if "vital_status.demographic" not in clinical.columns:
    raise KeyError("Column 'vital_status.demographic' not found in clinical file.")

clinical["dead"] = (clinical["vital_status.demographic"] == "Dead").astype(int)

# 2. Time: days_to_death for dead, last follow-up for alive
death_col = "days_to_death.demographic"
follow_col = "days_to_last_follow_up.diagnoses"

if death_col not in clinical.columns or follow_col not in clinical.columns:
    raise KeyError(
        "Required survival columns not found: "
        f"'{death_col}' and/or '{follow_col}'."
    )

# start with death time
clinical["time"] = clinical[death_col]

# replace with last follow-up for alive patients
mask_alive = clinical["dead"] == 0
clinical.loc[mask_alive, "time"] = clinical.loc[mask_alive, follow_col]

# convert to numeric, coerce bad values ('--', 'Not Reported', etc.)
clinical["time"] = pd.to_numeric(clinical["time"], errors="coerce")

# drop missing or non-positive times
clinical = clinical.dropna(subset=["time", "dead", "cancer_type"])
clinical = clinical[clinical["time"] > 0]

print(f"  -> Using {len(clinical)} patients with valid survival data.")

# Optionally convert time to years for nicer x-axis
clinical["time_years"] = clinical["time"] / 365.25

# ---------------- KAPLAN-MEIER ----------------
print("Generating Kaplan–Meier curves...")
plt.figure(figsize=(8, 6))
kmf = KaplanMeierFitter()

colors = {"LUAD": "tab:blue", "LUSC": "tab:orange"}

for subtype in ["LUAD", "LUSC"]:
    sub = clinical[clinical["cancer_type"] == subtype]
    if sub.empty:
        continue
    kmf.fit(
        durations=sub["time_years"],
        event_observed=sub["dead"],
        label=subtype,
    )
    kmf.plot_survival_function(
        ci_show=True,
        linewidth=2.5,
        color=colors.get(subtype, None),
    )

# ---------------- LOG-RANK TEST ----------------
luad_group = clinical[clinical["cancer_type"] == "LUAD"]
lusc_group = clinical[clinical["cancer_type"] == "LUSC"]

results = logrank_test(
    luad_group["time_years"], lusc_group["time_years"],
    event_observed_A=luad_group["dead"],
    event_observed_B=lusc_group["dead"],
)

p_val = results.p_value
print(f"  -> Log-rank p-value: {p_val:.5f}")

# ---------------- FINALIZE PLOT ----------------
plt.title(f"Overall Survival: LUAD vs LUSC (p = {p_val:.4f})",
          fontsize=14, fontweight="bold")
plt.xlabel("Time (years)", fontsize=12)
plt.ylabel("Survival probability", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(title="Subtype")
plt.tight_layout()

save_path = os.path.join(PLOTS_DIR, "survival_analysis.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Survival analysis complete. Plot saved to: {save_path}")
