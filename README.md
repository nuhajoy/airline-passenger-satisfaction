# ✈️ Airline Passenger Satisfaction Prediction

**Machine Learning Lab - Final Course Project**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 📋 Overview

**Problem:** Binary classification to predict airline passenger satisfaction  
**Target Variable:** `satisfaction` (satisfied vs neutral/dissatisfied)  
**Dataset:** 129,880 samples from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)  
**Best Model:** Random Forest Classifier with **96.34% accuracy** and **0.9939 ROC-AUC**

---

## 🚀 Quick Start

### 1. Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
Visit [Kaggle Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) and download `train.csv` and `test.csv` into the `data/` folder.

**Or use Git LFS:**
```bash
git lfs pull
```

### 3. Train Model
Open and run all cells in `notebooks/analysis.ipynb`:
```bash
jupyter notebook notebooks/analysis.ipynb
```

### 4. Launch Web App
```bash
streamlit run app.py
```
Access at: http://localhost:8501

---

## 📊 Dataset

- **Source:** Kaggle Airline Passenger Satisfaction
- **Training Set:** 103,904 samples
- **Test Set:** 25,976 samples
- **Features:** 22 (demographics, flight details, service ratings, delays)
- **Target Classes:** Satisfied (43.3%) vs Dissatisfied (56.7%)

**Key Features:**
- Demographics: Age, Gender, Customer Type
- Flight Info: Class, Type of Travel, Flight Distance
- Service Ratings: Wifi, Boarding, Entertainment, etc. (0-5 scale)
- Delays: Departure/Arrival delay minutes

---

## 🔧 Data Preprocessing

**Pipeline** (`src/preprocessing.py`):
1. **Missing Values:** Median imputation for Arrival Delay (0.3% missing)
2. **Encoding:**
   - Ordinal: Class (Eco → Eco Plus → Business)
   - One-Hot: Gender, Customer Type, Type of Travel
   - Label: Target variable
3. **Scaling:** StandardScaler for numerical features

---

## 🤖 Models & Results

### Baseline: Logistic Regression
- Accuracy: 87.12%
- ROC-AUC: 0.9256
- Training Time: ~6 seconds

### Advanced: Random Forest
- **Accuracy: 96.34%** (+9.22 pp improvement)
- **ROC-AUC: 0.9939** (+0.0683 improvement)
- **Precision: 0.9735** | **Recall: 0.9423** | **F1-Score: 0.9577**
- Training Time: ~45 seconds
- Hyperparameters: `n_estimators=200`, `max_depth=25`, `max_features='sqrt'`

### Error Reduction
- **71.6% fewer errors** (3,345 → 950 misclassifications)
- False Positives: 1,446 → 292 (80% reduction)
- False Negatives: 1,899 → 658 (65% reduction)

### Top Features (Random Forest)
1. Online Boarding (18.4%)
2. Inflight Wifi Service (15.2%)
3. Type of Travel - Business (9.9%)
4. Class - Business (8.3%)
5. Inflight Entertainment (7.2%)

**Insight:** Digital services explain 33.6% of satisfaction variance

---

## 🌐 Deployment

**Streamlit Web Application:**
- Real-time predictions with confidence scores
- Color-coded results (Satisfied ✅ / Dissatisfied ❌)
- Interactive UI for 12 key features
- Inference time: <100ms

**Run:** `streamlit run app.py`

---

## 📁 Project Structure

```
airline-passenger-satisfaction/
├── README.md                     # Project documentation
├── FINAL_PROJECT_REPORT.md       # Academic report
├── requirements.txt              # Dependencies
├── data/                         # Dataset (train.csv, test.csv)
├── src/
│   └── preprocessing.py          # Data preprocessing pipeline
├── notebooks/
│   └── analysis.ipynb            # EDA & modeling
├── models/                       # Trained models (.pkl)
└── app.py                        # Streamlit web app
```

---

## 🔁 Reproducibility

- ✅ Fixed random seed (`random_state=42`)
- ✅ Version-pinned dependencies
- ✅ Modular preprocessing pipeline
- ✅ Complete Git history
- ✅ Detailed documentation

---

## 📚 References

1. **Dataset:** Kaggle - Airline Passenger Satisfaction
2. **Framework:** scikit-learn, Streamlit, pandas
3. **Model:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32

---

## 📄 License

Educational use only - Machine Learning Lab coursework

---

**Status:** ✅ Complete  
**Last Updated:** January 31, 2026

🎉 Thank you for reviewing our project!
