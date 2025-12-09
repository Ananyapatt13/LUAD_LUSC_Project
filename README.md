# Lung Cancer Subtype Classification using TCGA RNA-seq Data

This project builds a complete and reproducible machine-learning pipeline for distinguishing **Lung Adenocarcinoma (LUAD)** from **Lung Squamous Cell Carcinoma (LUSC)** using transcriptomic (RNA-seq) data from **The Cancer Genome Atlas (TCGA)**.

The workflow includes:

- Data preparation  
- Log-transformation and normalization  
- Machine-learning classification using a linear SVM  
- Cross-validation and model benchmarking  
- Gene-level biological interpretation  
- Functional enrichment analysis  
- Kaplan–Meier survival analysis  

The project is structured so that **any beginner can understand**, yet follows best practices used in real research pipelines.

----

## ⭐ Key Results

- **Accuracy (Test Set):** ~96%  
- **Cross-Validation Accuracy:** ~95%  
- **AUC:** ~0.99  
- **Biological Markers Identified:**  
  - LUSC: keratins and squamous-differentiation genes  
  - LUAD: lineage and secretory-cell markers  
- **Enrichment:**  
  - LUAD → developmental & signaling pathways  
  - LUSC → epidermal, keratinization, and cell-structure programs  
- **Survival:** No significant difference (log-rank p ≈ 0.24)

---

## 📂 Project Structure

LUAD_LUSC_Project/
│
├── data/
│ ├── TCGA-LUAD.clinical.tsv
│ ├── TCGA-LUSC.clinical.tsv
│ ├── TCGA-LUAD.star_tpm.tsv
│ ├── TCGA-LUSC.star_tpm.tsv
│ ├── expression.csv
│ └── labels.csv
│
├── scripts/
│ ├── prepare_data.py
│ ├── train_svm.py
│ ├── evaluate_svm.py
│ ├── cross_validation.py
│ ├── model_comparision.py
│ ├── enrichment_analysis.py
│ └── survival_analysis.py
│
├── results/
│ ├── svm_model.pkl
│ ├── scaler.pkl
│ ├── variance_selector.pkl
│ ├── top_genes.csv
│ ├── enrichment_LUAD.csv
│ └── enrichment_LUSC.csv
│
└── plots/
├── pca.png
├── roc.png
├── confusion_matrix.png
├── heatmap.png
├── enrichment_LUAD.png
├── enrichment_LUSC.png
└── survival_analysis.png


---

## 🧪 Methods Overview (Beginner-Friendly)

### **1. Data Preparation**
- TCGA RNA-seq TPM matrices were merged.
- Clinical files were loaded and mapped.
- A final `expression.csv` matrix (samples × genes) and `labels.csv` (sample → subtype) were created.

### **2. Log Transformation**
RNA-seq data is right-skewed.  
We apply:



log1p((x) = log(x + 1)


This prevents extreme outliers from dominating PCA or SVM decision boundaries.

### **3. Train/Test Split**
- 80% Train  
- 20% Test  
- Stratified so LUAD/LUSC representation is preserved.

### **4. Feature Selection**
Low-variance genes removed using:

VarianceThreshold(0.1)


This focuses the model on biologically informative genes.

### **5. Scaling**
Z-score standardization is fit **only on the training data** to avoid data leakage.

### **6. Model Training**
A **linear SVM** is used:
- Excellent for high-dimensional gene expression data  
- Coefficients directly show top predictive genes  

The trained model and preprocessing objects are saved as `.pkl` files.

### **7. Evaluation**
The following are generated:
- Confusion matrix  
- PCA visualization  
- ROC curve  
- Heatmap of top discriminatory genes  

### **8. Biological Enrichment**
Top LUAD and LUSC genes (based on SVM weights) are analyzed using:

- **Enrichr (GSEApy)**  
- **GO Biological Processes 2021**

### **9. Survival Analysis**
Using TCGA clinical metadata:
- Kaplan–Meier curves  
- Log-rank statistical testing  

Result: LUAD and LUSC show **no significant difference** in overall survival in this dataset.

---

## 📦 Installation

pip install numpy pandas scikit-learn matplotlib seaborn gseapy lifelines mygene


---

## ▶️ How to Run the Pipeline

### **1. Prepare the Data**
python scripts/prepare_data.py

### **2. Train the SVM**
python scripts/train_svm.py

### **3. Evaluate**
python scripts/evaluate_svm.py

### **4. Cross-Validation**
python scripts/cross_validation.py


### **5. Benchmark Models**
python scripts/model_comparision.py


### **6. Perform Enrichment Analysis**
python scripts/enrichment_analysis.py

### **7. Run Survival Analysis**
python scripts/survival_analysis.py

---

## 📘 What This Project Demonstrates

- How raw RNA-seq TPM counts can be transformed into machine-learning-ready features.
- How to prevent data leakage and build a scientifically valid pipeline.
- How classical ML (linear SVM) performs extremely well on high-dimensional biomedical data.
- How predictive gene signatures can be interpreted biologically.
- How computational oncology integrates molecular, clinical, and statistical layers.

---

## 📝 Future Improvements

- Add external validation using GEO datasets.  
- Integrate more cancer subtypes (pan-cancer classification).  
- Test nonlinear models (XGBoost, deep learning).  
- Build a small app to classify new samples based on TPM values.

---

## 🤝 Acknowledgements

- Data sourced from **The Cancer Genome Atlas (TCGA)**.  
- Gene Ontology enrichment via **Enrichr / GSEApy**.  
- Survival analysis via **lifelines**.  

---

## 📄 License

MIT License 




