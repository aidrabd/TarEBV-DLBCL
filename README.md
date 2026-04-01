# 🧬 TarEBV-DLBCL (Targeting EBV-associated Diffuse Large B-Cell Lymphoma (DLBCL))
A **Multi-Omics & Machine Learning–Driven Drug Discovery Framework** for identifying **Natural Product inhibitors** against **EBV-associated Diffuse Large B-Cell Lymphoma (DLBCL)**

---

## 🚀 Key Applications
- **Natural Drug Discovery:** Identify bioactive compounds from natural resources with potential anticancer and/or antiviral potential
- **Multi-Target Screening:** Explore multi-target inhibition of critical oncogenic proteins (AKT1, BCL2)
- **Lead Proritization:** Prioritize compounds for in vitro and in vivo validation

---

## 📊 Model Performance
The best model: **Random Forest regression model** demonstrated strong predictive performance with excellent generalization ability:  

| Dataset | R² Score of AKT1 | R² Score of BCL2 | 
|----------|-----------|-----------|
| Training Set | 0.928 | 0.937 |
| Validation Set | 0.894 | 0.911 |
| Cross Validation Set | **0.8634 ± 0.0158** | **0.8965 ± 0.0146** |

The high overall R² highlights the robustness of the model in predicting inhibitory activity (pIC₅₀).  

---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.8  
- `conda` package manager  

### Install
### Prerequisites

- Ubuntu/Linux terminal/Window Command Line

### Installation

```bash
# Clone and setup
git clone https://github.com/aidrabd/TarEBV-DLBCL.git
cd TarEBV-DLBCL

# Make prediction script executable
chmod +x predict.py
```

First, make sure you have conda installed:

```bash
1. Install  Miniconda (if not installed)

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

2. Activate conda base environment

Type "conda init"

Then, restart your terminal

After that, activate the base environment with:

conda activate
```

Second, make sure you have Python specific version installed:

```bash
conda create -n py312 python=3.12.9
conda activate py312
python --version
```

---

## 🏃‍♂️ Simple Start

### Command Line Usage
```bash
python predict.py
 python predict.py --models akt1.h5 --input top4_compounds_of_Ganoderma.csv --output out.csv
 python predict.py --models bcl2.h5 --input top4_compounds_of_Ganoderma.csv --output out.csv
# Provide your input file instead of top4_compounds_of_Ganoderma.csv
# Columns required in File: SMILES, pIC50 (Leave pIC50 empty for prediction)
```

### 🧾 Input Format
| Column | Description |
|---------|--------------|
| **SMILES** | Simplified Molecular Input Line Entry System notation |
| **pIC50** | Predicted biological activity (keep empty for prediction) |
# See top4_compounds_of_Ganoderma.csv file to prepare your own file.
---

## 📖 Scientific Background
**TarEBV-DLBCL** study is designed to accelerate *in silico* screening of natural anticancer and/or antiviral agents targeting both AKT1 and BCL2 by integrating:  
- **Machine Learning** for predictive modeling  
- **Molecular Docking** for binding affinity evaluation  
- **Molecular Dynamics Simulations (200 ns)** for stability validation  

---

## ⚠️ Disclaimer
**TarEBV-DLBCL** is developed for **research purposes only**.  
All predictions should be validated experimentally before any clinical or commercial application.  

---

## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.  

---

## 📚 Citation
If you use **TarEBV-DLBCL** in your research, please cite:  

---

## 🙏 Acknowledgments
- Training data curated from **ChEMBL** and published literature  
- Molecular descriptors generated using **RDKit**   
- Special thanks to the **open-source bioinformatics and cheminformatics communities**
