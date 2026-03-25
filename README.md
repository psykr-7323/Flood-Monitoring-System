# 🌊 Active Flood Risk Management System

An advanced, machine-learning-driven dashboard designed to calculate real-time flood probabilities based on complex environmental and infrastructural factors. This project was developed as part of an Advanced Data Laboratory and demonstrates comprehensive data preprocessing, feature engineering, and comparative algorithmic analysis.

## 📌 Project Overview
Predicting flood events requires analyzing vast amounts of interconnected data. This system processes a dataset of over 1.1 million records, mathematically aggregating 20 raw environmental variables into **7 Master Indices**. These indices are then fed into a highly optimized machine learning pipeline to generate accurate, real-time risk assessments.

The final output is presented in an interactive **Streamlit Dashboard** that allows stakeholders to tweak environmental conditions and immediately see the predicted impact, alongside a transparent breakdown of model evaluation metrics.

---

## 🧬 Feature Engineering: The 7 Master Indices
To reduce dimensionality and optimize system performance, the original dataset features were logically grouped and averaged into the following core indices:

1. **Climate Risk:** Combines *Monsoon Intensity*, *Climate Change*, and *Coastal Vulnerability*.
2. **Geological Risk:** Combines *Topography Drainage*, *Landslides*, and *Watersheds*.
3. **Human Impact:** Combines *Deforestation* and *Urbanization*.
4. **Infrastructure Deficit:** Combines *Dams Quality*, *Drainage Systems*, and *Deteriorating Infrastructure*.
5. **Environmental Degradation:** Combines *Siltation*, *Agricultural Practices*, and *Wetland Loss*.
6. **Management Failures:** Combines *River Management*, *Ineffective Disaster Preparedness*, and *Inadequate Planning*.
7. **Encroachment Level:** The physical degree of construction on natural waterways.

*(Note: Population and Political Factors were intentionally removed to isolate the strictly physical/environmental drivers of flooding).*

---

## 🤖 Machine Learning Architecture
The system evaluates real-time data using three distinct algorithms to provide comparative mathematical verification:

* **Linear Regression:** Serves as the parametric linear baseline.
* **Decision Tree Regressor:** A non-parametric model used to capture non-linear thresholds and extract explicit Feature Dependency/Importance.
* **Multi-Layer Perceptron (MLP Neural Network):** The primary engine. By processing the engineered indices through hidden layers, the Perceptron successfully maps deep, non-linear relationships to achieve the highest predictive accuracy.

---

## 📂 Repository Structure

* `app.py`: The interactive Streamlit web application and primary user interface.
* `Model_Training.ipynb`: The core Jupyter Notebook containing Exploratory Data Analysis (EDA), feature engineering, model training, and advanced statistical evaluation (R-squared, MSE, RMSE, MAE).
* `train.csv`: The primary dataset used for training and validation.
* `scaler.pkl`: The serialized Standard Scaler used to normalize data for the Neural Network.
* `linear_model.pkl`: Serialized Linear Regression model.
* `dt_model.pkl`: Serialized Decision Tree model.
* `mlp_model.pkl`: Serialized Multi-Layer Perceptron model.

---

## 🚀 Installation & Usage

# 📦 Prerequisites

Ensure you have **Python 3.8+** installed.

Install the required libraries:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib streamlit joblib
```

---

# 🚀 Running the Dashboard

1. Clone this repository to your local machine.

2. Open your terminal or command prompt and navigate to the repository folder.

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

4. The dashboard will automatically open in your browser  
   (typically at **http://localhost:8501**).

---

# 🔬 Exploring the Data Science Pipeline

To review the mathematical foundation of the project:

- Open `Model_Training.ipynb` in **Jupyter Notebook** or **VS Code**

The notebook is organized into the following steps:

1. **Data Loading & Feature Engineering**
2. **EDA (Correlation Heatmaps & Scatter Plots)**
3. **Model Training & Serialization**
4. **Feature Importance Extraction**
5. **Holdout Evaluation**

---

# 📊 Dashboard Features

### 🎛️ Interactive Inputs
- Adjust the **7 Master Indices**  
- Range: **0.0 to 20.0**
- Simulate different environmental scenarios

### ⚡ Live Predictions
- Instantly view the predicted **Flood Probability**

### 🚨 Alert System
- Automatically classifies risk levels:
  - **Low Risk**
  - **Moderate Risk**
  - **High Risk**
- Provides actionable recommendations

### 📈 Metrics Expander
- Includes:
  - Data dictionary
  - Transparent model evaluation metrics
- Helps validate the system’s mathematical credibility
