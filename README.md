# TCGA LUAD–LUSC Classification Using Machine Learning and Transcriptomics
A Complete Bioinformatics & ML Pipeline for Lung Cancer Subtype Classification

This repository presents a fully reproducible, end-to-end computational oncology pipeline to classify Lung Adenocarcinoma (LUAD) vs Lung Squamous Cell Carcinoma (LUSC) using RNA-Seq gene expression data from The Cancer Genome Atlas (TCGA).

The workflow integrates:
- High-dimensional gene expression analysis
- Dimensionality reduction (PCA)
- Machine Learning (Linear SVM)
- Gene-level feature interpretation
- Functional enrichment (GO/KEGG)
- Survival analysis using clinical metadata

This repository is structured in an educational and research-ready manner, suitable for:
- Bioinformatics students
- Computational biology researchers
- ML practitioners entering genomics
- Thesis / dissertation projects
- Reproducibility and benchmarking

------------------------------------------------------------

# Project Highlights

Scientific Goal  
Distinguish two major NSCLC (non–small cell lung cancer) subtypes using transcriptomics.

Machine Learning Goal  
Create a robust SVM model capable of subtype prediction with >95% accuracy.

Bioinformatics Goals  
Identify subtype-specific genes and pathways, and evaluate their clinical relevance.

Key Achievements  

| Task                            | Result                                                                     |
| ------------------------------- | -------------------------------------------------------------------------- |
| SVM Classification Accuracy     | ~96%                                                                       |
| AUC (ROC)                       | ~0.99                                                                      |
| Dimensionality Reduction        | PCA separates LUAD and LUSC clearly                                        |
| Biological Signals              | LUAD enriched for developmental pathways; LUSC enriched for keratinization |
| Survival Analysis               | No significant survival difference (log-rank p ≈ 0.24)                     |

------------------------------------------------------------

# Conceptual Workflow

Raw TCGA Data → Preprocessing → Feature Engineering → PCA → SVM Model → Gene Interpretation → Enrichment → Survival Analysis

------------------------------------------------------------

# Directory Structure

```

LUAD_LUSC_Project/
│
├── data/
│   ├── TCGA-LUAD.clinical.tsv
│   ├── TCGA-LUSC.clinical.tsv
│   └── labels.csv
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_svm.py
│   ├── evaluate_svm.py
│   ├── cross_validation.py
│   ├── model_comparision.py
│   ├── enrichment_analysis.py
│   └── survival_analysis.py
│
├── results/
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── variance_selector.pkl
│   ├── enrichment_LUAD.csv
│   ├── enrichment_LUSC.csv
│   └── top_genes.csv
│
├── plots/
│   ├── pca.png
│   ├── roc.png
│   ├── confusion_matrix.png
│   ├── heatmap.png
│   ├── enrichment_LUAD.png
│   ├── enrichment_LUSC.png
│   └── survival_analysis.png
│
└── README.md

```

------------------------------------------------------------

# Methodology (Detailed)

## 1. Data Acquisition & Preprocessing

TCGA RNA-Seq TPM matrices (not included due to size and licensing) were processed to create:
- A consolidated LUAD/LUSC label file
- Clinical metadata tables
- Final expression matrix (external due to size)

Processing steps include:
```

X_log = log2(X + 1)

```
- Filtering low-expressed genes
- Aligning clinical labels
- Merging LUAD and LUSC cohorts

------------------------------------------------------------

## 2. Feature Engineering

Variance Thresholding  
Genes with near-constant expression were removed:
```

VarianceThreshold(threshold=0.1)

```

Standard Scaling  
Fit on training data only:
```

Z = (X - μ) / σ

```

------------------------------------------------------------

## 3. PCA

- Used to visualize LUAD–LUSC separation  
- PC1 vs PC2 shows clear subtype separation  
Plot: `plots/pca.png`

------------------------------------------------------------

## 4. SVM Classification

Linear SVM pipeline:
- 80/20 train–test split
- StandardScaler
- Linear SVM
- Confusion matrix, ROC, AUC

Results:
```

Accuracy ≈ 96%
AUC ≈ 0.99

```

Artifacts:
- `svm_model.pkl`
- `scaler.pkl`
- `top_genes.csv`

------------------------------------------------------------

## 5. Biological Interpretation

Top genes extracted from SVM coefficients.

LUAD-specific genes:
- Developmental and signaling pathways

LUSC-specific genes:
- Keratinization
- Epithelial differentiation

Heatmap: `plots/heatmap.png`

------------------------------------------------------------

## 6. Functional Enrichment (GO/KEGG)

Performed using GSEApy (Enrichr API).

LUAD:
- MAPK signaling
- Cell differentiation
- Developmental pathways

LUSC:
- Keratinization
- Epithelial differentiation
- Structural organization

Results:
- `results/enrichment_LUAD.csv`
- `results/enrichment_LUSC.csv`

Plots:
- `plots/enrichment_LUAD.png`
- `plots/enrichment_LUSC.png`

------------------------------------------------------------

## 7. Survival Analysis

Using lifelines:
- Kaplan–Meier curves
- Log-rank test

Outcome:
- No significant survival difference (p ≈ 0.24)

Plot: `plots/survival_analysis.png`

------------------------------------------------------------

# How to Reproduce

Install dependencies:
```

pip install numpy pandas scikit-learn seaborn matplotlib gseapy lifelines mygene

```

Preprocessing:
```

python scripts/prepare_data.py

```

Train SVM:
```

python scripts/train_svm.py

```

Evaluate:
```

python scripts/evaluate_svm.py

```

Enrichment:
```

python scripts/enrichment_analysis.py

```

Survival:
```

python scripts/survival_analysis.py

```

------------------------------------------------------------

# Data Acquisition (Direct TCGA Links)

Due to size and redistribution restrictions, raw datasets are not included.

Download data from the official GDC-hosted files:

LUAD TPM  
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.star_tpm.tsv.gz

LUSC TPM  
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUSC.star_tpm.tsv.gz

LUAD Clinical  
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.clinical.tsv.gz

LUSC Clinical  
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUSC.clinical.tsv.gz

Place downloaded files into:
```

LUAD_LUSC_Project/data/

```

------------------------------------------------------------

# Future Improvements

- Multi-cancer classification (pan-cancer ML)
- Add nonlinear models (XGBoost, Deep Learning)
- SHAP / LIME explainability
- Build an interactive classification web app
- Apply batch correction (ComBat)

------------------------------------------------------------

# Acknowledgments

- TCGA Research Network
- Enrichr / GSEApy
- Lifelines Python package
- SVM theory: Cortes & Vapnik (1995)

------------------------------------------------------------

# Author

Ananya Pattjoshi  
Bioinformatics • Machine Learning • Genomics  
GitHub: https://github.com/Ananyapatt13
