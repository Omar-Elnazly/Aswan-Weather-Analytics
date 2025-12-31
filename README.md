# 🌦️ Aswan Weather Data Analysis & Machine Learning

---

**Notebook Version by my pair Ezzeldin Salah:**  
[https://github.com/huevvn/SolarPowerPredictionPhase2]

This notebook presents a comprehensive pipeline for **solar power prediction using meteorological data**, including data cleaning, extensive feature engineering, statistical analysis, and machine learning modeling for both **classification and regression**.

---

## 📌 Project
Overview
Environmental and climate analysis plays a critical role in understanding regional weather patterns and supporting decision-making in agriculture, renewable energy, and urban planning.

This project analyzes **Aswan weather data (Egypt)** using **statistical analysis, data visualization, feature engineering, dimensionality reduction, and multiple machine learning models**.  
The primary goals are:
- Understand climate behavior in one of Egypt’s hottest regions
- Analyze relationships between meteorological variables
- Classify temperature levels
- Predict solar power generation
- Compare multiple ML models and evaluate overfitting

---

## 📊 Dataset Description
- **Location:** Aswan, Egypt  
- **Type:** Time-series weather dataset  
- **Source:** Egyptian Meteorological Authority (2022)

### Main Features
| Feature | Description |
|------|-----------|
| AvgTemperature (°C) | Daily average temperature |
| Humidity (%) | Relative humidity |
| Wind (m/s) | Average wind speed |
| Pressure (hPa) | Atmospheric pressure |
| Solar (PV) | Solar power indicator |

### Engineered Features
- `Temp_bin`: Low / Medium / High temperature classes  
- `Humidity_bin`: Low / Medium / High humidity classes  

---

## 🔍 Exploratory Data Analysis (EDA)
- Dataset inspection and sampling
- Missing value checks
- Column removal (`Unnamed`, `Date`)
- Temperature & humidity binning
- Summary statistics
- Distribution analysis

---

## 📈 Statistical Analysis
- Summary statistics (mean, variance, skewness, kurtosis)
- Covariance matrix + heatmap
- Correlation matrix + heatmap
- **Hypothesis Testing**
  - Independent **T-Test**
  - **ANOVA**
  - **Chi-Square Test**
- Correlation insights between weather variables

---

## 📉 Data Visualization
- Temperature distribution
- Temperature vs humidity scatter plots
- Temperature across humidity levels (boxplots)
- Correlation heatmaps

---

## 🧠 Feature Engineering
- Train-test split (80% / 20%)
- Standardization using `StandardScaler`

---

## 🤖 Machine Learning Models

### 🔹 Dimensionality Reduction
- **PCA** (Principal Component Analysis)
- **LDA** (Linear Discriminant Analysis)
- **SVD** (Singular Value Decomposition)

### 🔹 Classification Models
- Naive Bayes (+ Cross Validation)
- Decision Tree (Entropy)
- K-Nearest Neighbors (Euclidean, Manhattan, Chebyshev)
- Logistic Regression
- Neural Network (Feed Forward)
- Neural Network (Feed Back – custom implementation)
- Bayesian Belief Network

### 🔹 Regression Model
- Linear Regression for **Solar(PV)** prediction

---

## 📊 Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Cross-validation
- ROC Curves (multi-class)
- Regression metrics:
  - MAE
  - RMSE
  - R²
  - Willmott’s Index
  - Nash–Sutcliffe Efficiency
  - Legates–McCabe Index

---

## ⚠️ Overfitting Analysis
Comparison between training and testing accuracy for:
- Naive Bayes
- Decision Tree
- LDA
- KNN

📌 **Result:**  
Most models demonstrate **good fit** with minimal overfitting.

---

## 🏆 Key Results
- **Best Classification Accuracy:** Decision Tree (100%)
- **Strongest Correlations:**
  - Temperature ↔ Dew Point
  - Humidity ↔ Pressure
- Solar power prediction remains challenging due to nonlinear relationships

---

## 📚 Related Work
- Li et al. (2020): Machine Learning for Solar Power Forecasting
- Egyptian Meteorological Authority (2022): Official Aswan weather data

This project aligns with prior studies by demonstrating the effectiveness of ML models in weather and renewable energy analysis.

---

## 🛠️ Technologies & Libraries
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- SciPy

---

## 📌 How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy
3. Open the Jupyter Notebook
4. Run all cells sequentially

---

## 📄 References

1. Li, X., Zhang, Q., & Wang, Y. (2020). *Machine Learning Approaches for Solar Power Forecasting*. Renewable Energy.
2. Egyptian Meteorological Authority. (2022). *Aswan Weather Dataset*.

---

## 👤 Author

**Omar Sayed**  
Computer Science Student – AI & Data Science
