# IPO Quantitative Analysis and Listing Predictor

A machine learning pipeline that predicts IPO listing gain percentage as a continuous regression output, trained on 800+ SEBI prospectuses. The project combines financial feature engineering, text extraction from PDFs, and ensemble modeling to estimate how much an IPO will gain (or lose) on listing day.

---

## Overview

Most retail IPO analysis is vibes-based — GMP trackers and subscription data repackaged as conviction. This project tries to do it properly: extract structured financial data from SEBI prospectuses, engineer quantitative features, and train regression models on actual listing outcomes.

The output is a predicted listing gain percentage. Not a buy/sell signal, just a number the model thinks is likely given what's in the filings.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data extraction | `pdfplumber`, `PyMuPDF` |
| Financial data | `yfinance` |
| Feature engineering | `pandas`, `numpy` |
| Modeling | `XGBoost`, `Random Forest` (scikit-learn) |
| Environment | Google Colab, Python 3.10+ |

---

## Features

**Engineered features used in the model:**

- `Subscription_Growth_Rate` — momentum in subscription demand across tranches
- `Demand_Convexity` — non-linearity in QIB/HNI/Retail subscription spread
- `Past_1_Month_Index_Return` — market regime at the time of listing
- `Issue_Size` — offering size as a demand/supply signal
- `GMP_Proxy` (where available) — grey market premium as a sentiment feature

**Features dropped during cleaning:**
- `Issue_Size_Norm`, `PE_Ratio`, and a few others were removed due to unit corruption across filings (mixed crore/lakh/million formatting with no reliable normalization).

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/IPO-QUANTITATIVE-ANALYSIS-AND-LISTING-PREDICTOR.git
cd IPO-QUANTITATIVE-ANALYSIS-AND-LISTING-PREDICTOR

# Install dependencies
pip install -r requirements.txt
```

For Colab, mount your Drive and set the data path:

```python
from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/LISTING PRICE FIX/'
```

---

## Project Structure

```
├── data/
│   └── ipo_dataset_cleaned.csv       # Cleaned dataset (800+ IPOs)
├── notebooks/
│   ├── 01_data_extraction.ipynb      # PDF scraping + field parsing
│   ├── 02_feature_engineering.ipynb  # Feature construction
│   └── 03_modeling.ipynb             # XGBoost / RF training + eval
├── src/
│   ├── extract.py                    # pdfplumber/PyMuPDF extraction logic
│   ├── features.py                   # Feature engineering functions
│   └── model.py                      # Training pipeline
├── requirements.txt
└── README.md
```

---

## Results

Models trained and evaluated on an 80/20 train-test split:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost | ~18.4% | ~12.1% | 0.41 |
| Random Forest | ~19.7% | ~13.3% | 0.37 |

> Note: IPO listing gains are noisy by nature — sentiment, market conditions, and last-minute GMP shifts explain a lot of variance that structured data can't fully capture. These numbers are directionally useful, not oracle-level accurate.

**Most predictive features (by XGBoost importance):**
1. `Past_1_Month_Index_Return`
2. `Subscription_Growth_Rate`
3. `Demand_Convexity`

---

## Limitations

- Data quality is uneven across older filings — some fields required manual correction
- GMP data was not available for all IPOs, so it's an optional feature
- The model is trained on Indian IPOs listed on NSE/BSE; it won't generalize to other markets
- `PE_Ratio` and a few size normalization columns were dropped due to inconsistent units across prospectuses — this is a known gap

---

## Author

**Divy Gupta**  
B.Tech DSEB, Plaksha University (Batch 2024–28)  
[LinkedIn](https://www.linkedin.com/in/divy-gupta-61720725a/) 

**Budati Nagaveni**  
B.Tech DSEB, Plaksha University (Batch 2024–28)  
[LinkedIn_Nagaveni](https://www.linkedin.com/in/budati-nagaveni-20b49235b/) 
