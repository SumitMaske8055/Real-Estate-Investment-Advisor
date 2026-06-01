Here's a complete README for your project:

---

# 🏠 Real Estate Investment Advisor

A machine learning-powered web application built with **Streamlit** that predicts whether a property is a **good investment** and estimates its **future value after 5 years**, trained on 250,000 Indian housing records.

---

## 🚀 Features

- 📊 Predicts if a property is a **Good Investment** (Classification)
- 💰 Estimates **5-Year Future Price** in Lakhs (Regression)
- 🎯 Confidence score for investment decisions
- 🖥️ Interactive **Streamlit web app** for real-time predictions
- 🏙️ Trained on pan-India data across multiple states and cities

---

## 🧠 Models Used

| Task | Model | Performance |
|------|-------|-------------|
| Classification (Good Investment?) | Random Forest | Accuracy: 99.92% \| F1: 99.93% |
| Classification (Baseline) | Logistic Regression | Accuracy: 87.37% |
| Regression (Future Price) | Random Forest | RMSE: 3.37L \| R²: 0.9997 |
| Regression (Baseline) | Ridge Regression | RMSE: 145.48L \| R²: 0.5036 |

✅ **Best Models: Random Forest (both tasks)**

---

## 📁 Project Structure

```
real-estate-investment-advisor/
│
├── app.py                          # Streamlit web application
├── Real_Estate_Investment_Advisor.ipynb  # Model training notebook
│
├── models/
│   ├── clf_rf_model.pkl            # Random Forest Classifier
│   ├── reg_rf_model.pkl            # Random Forest Regressor
│   ├── label_encoders.pkl          # Label Encoders
│   └── scaler.pkl                  # Standard Scaler
│
├── data/
│   └── india_housing_prices.csv    # Dataset (250,000 records)
│
└── requirements.txt
```

---

## 📦 Dataset

- **Source:** `india_housing_prices.csv`
- **Size:** 250,000 property records | 23 features
- **Coverage:** Multiple Indian states and cities
- **Key Features:** BHK, Size (SqFt), Price (Lakhs), Year Built, Amenities, Transport Accessibility, Security, Parking, and more

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/real-estate-investment-advisor.git
cd real-estate-investment-advisor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 🔮 How It Works

1. User inputs property details (location, size, BHK, amenities, etc.)
2. Inputs are preprocessed using saved **Label Encoders** and **Scaler**
3. **Random Forest Classifier** predicts if it's a good investment + confidence %
4. **Random Forest Regressor** predicts the estimated price after 5 years
5. Results are displayed instantly on the Streamlit dashboard

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Scikit-learn** — ML models
- **Streamlit** — Web application
- **Pandas & NumPy** — Data processing
- **Matplotlib & Seaborn** — EDA visualizations
- **Joblib/Pickle** — Model serialization

---

## 📊 Results

```
Dataset         : 250,000 properties | 29 features
Good Investments: 134,349 (53.7%)
Avg Future Price: ₹374.1 Lakhs (8% p.a. × 5 years)

CLASSIFICATION → Random Forest: Accuracy 0.9992 | F1 0.9993
REGRESSION     → Random Forest: RMSE 3.37L | MAE 1.99L | R² 0.9997
```

---

## 🙌 Acknowledgements

- Dataset: Indian Housing Prices (Synthetic/Public)
- Built as part of a real estate analytics and ML portfolio project
