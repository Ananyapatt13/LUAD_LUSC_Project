import os
import pandas as pd

# Load the file to inspect headers
path = "../data/TCGA-LUAD.clinical.tsv"

print(f"Inspecting: {path}")
try:
    df = pd.read_csv(path, sep="\t")
    print("\n--- FOUND COLUMNS ---")
    for col in df.columns:
        print(col)
except Exception as e:
    print(e)