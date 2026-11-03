# 🌋 Landslide Susceptibility Classification

A machine learning-based web application for **Landslide Susceptibility Classification** using **Random Forest and Logistic Regression**. The project analyzes environmental and soil-related factors and provides an interactive prediction through a **Streamlit web application**.

---


---

## 📌 Project Overview

Landslides are natural hazards influenced by several environmental and geological factors such as rainfall, slope, soil saturation, vegetation cover, earthquake activity, and proximity to water.

This project applies supervised machine learning algorithms to classify whether a given set of environmental conditions is associated with a **landslide or no landslide**.

The included dataset is a reproducible simulation based on plausible environmental
relationships. It is intentionally noisy and overlapping so that statistical model
comparison is meaningful. It must not be presented as a replacement for a validated
geological field dataset.

The project compares two machine learning algorithms:

- 🌳 Random Forest
- 📊 Logistic Regression

The trained models are integrated into a **Streamlit application**, allowing users to enter environmental conditions and obtain a prediction interactively.

---

## 🎯 Objectives

- Analyze environmental factors associated with landslides.
- Perform basic exploratory data analysis.
- Preprocess the dataset for machine learning.
- Handle infinite values, missing targets, duplicate rows, and missing features.
- Add exactly 1,200 reproducible SMOTE samples to the training split only.
- Train multiple classification algorithms.
- Compare model performance using standard evaluation metrics.
- Use stratified five-fold cross-validation to report mean accuracy and variation.
- Estimate the accuracy difference with bootstrap confidence intervals.
- Test the paired accuracy difference with a reproducible permutation test.
- Visualize confusion matrices and ROC curves.
- Analyze Random Forest feature importance.
- Build an interactive Streamlit prediction application.

---

## 📂 Dataset

The application uses the supplied `landslide_dataset.csv` containing **3,200
original observations** and the following features. The separate
`landslide_datasetS.csv` file is not used.

| Feature | Description |
|---|---|
| `Rainfall_mm` | Rainfall measurement |
| `Slope_Angle` | Slope angle of the area |
| `Soil_Saturation` | Level of soil saturation |
| `Vegetation_Cover` | Vegetation coverage |
| `Earthquake_Activity` | Earthquake activity level |
| `Proximity_to_Water` | Proximity to nearby water sources |
| `Landslide` | Target variable: 0 = No Landslide, 1 = Landslide |
| `Soil_Type_Gravel` | One-hot encoded gravel soil type |
| `Soil_Type_Sand` | One-hot encoded sand soil type |
| `Soil_Type_Silt` | One-hot encoded silt soil type |

The generator creates correlated rainfall, slope, soil saturation, vegetation,
earthquake, water-distance, and soil-type variables. Labels are generated from a
nonlinear susceptibility score with modest random noise. Approximately 1% of each
feature column is missing intentionally so the imputation pipeline is exercised.
Generate a separate 2,000-row example dataset with:

```bash
python generate_landslide_dataset.py
```

This creates `generated_landslide_dataset.csv` and does not overwrite the
supplied `landslide_dataset.csv`.

---

## 🧠 Machine Learning Models

### 1. Random Forest

A tree-based ensemble learning algorithm consisting of multiple decision trees.

Configuration:

```text
n_estimators = 300
max_depth = 8
min_samples_leaf = 5
max_features = sqrt
class_weight = balanced
random_state = 42
```

Random Forest is also used to calculate feature importance.

### 2. Logistic Regression

A linear classification algorithm used as a baseline model.

The features are scaled using `RobustScaler`, which uses medians and interquartile
ranges and is less sensitive to extreme observations.

```text
C = 0.3
max_iter = 1000
random_state = 42
```

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Infinite Value, Missing Value & Duplicate Handling
   ↓
Exploratory Data Analysis
   ↓
Feature / Target Separation
↓
Train-Test Split
↓
Train-only SMOTE (+1,200 synthetic rows; 3,760 effective training rows)
↓
Model Training
├── Random Forest
└── Logistic Regression
   ↓
Model Evaluation
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1 Score
   ├── ROC-AUC
   ├── 5-fold CV Accuracy Mean
   └── 5-fold CV Accuracy Std
   ↓
Visualization
   ├── Confusion Matrix
   ├── ROC Curve
   └── Feature Importance
   ↓
Streamlit Application
   ↓
Landslide Prediction
```

---

## 📊 Model Evaluation

The models are evaluated using:

### Accuracy

Measures the percentage of correctly classified observations.

### Precision

Measures how many predicted landslide cases were actually landslides.

### Recall

Measures how many actual landslide cases were correctly identified.

### F1 Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Statistical comparison

Logistic Regression is treated as the baseline and Random Forest as the candidate improved model. Both models predict the same held-out test rows. The application reports:

- The observed Random Forest minus Logistic Regression accuracy difference.
- A 95% bootstrap confidence interval using 5,000 resamples.
- A paired permutation-test p-value using 10,000 sign permutations.

At the 5% significance level, an interval containing zero or a p-value of at least 0.05 means the observed improvement is not conclusive evidence of better population performance. Model accuracy intervals are also estimated with 5,000 bootstrap resamples.

The supplied simulated dataset is designed to avoid perfect separability. Its
performance difference is still an estimate for this generated population, not
evidence about real geographic regions. For a real deployment, replace it with
independently collected and georeferenced observations.

---

## 📈 Streamlit Application Features

The application provides several interactive sections.

### 📊 Dashboard

Displays:

- Total number of samples
- Number of features
- Landslide samples
- Non-landslide samples
- Dataset preview
- Target distribution
- Correlation matrix

### 🤖 Model Comparison

Displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Five-fold cross-validation accuracy mean and standard deviation
- Classification reports
- Model comparison chart

### 📐 Statistical Evaluation

Compares Random Forest with the Logistic Regression baseline using bootstrap sampling and a paired permutation hypothesis test on the shared test set.

### 🔲 Confusion Matrices

Interactive confusion matrices are available for:

- Random Forest
- Logistic Regression

### 📈 ROC Curves

The application displays ROC curves for both models and their corresponding AUC values.

### 🌳 Feature Importance

Random Forest feature importance is displayed as a table and bar chart.

### 🔮 Prediction

Users can enter:

- Rainfall
- Slope angle
- Soil saturation
- Vegetation cover
- Earthquake activity
- Proximity to water
- Soil type

The selected model then predicts:

```text
LANDSLIDE
```

or

```text
NO LANDSLIDE
```

along with the estimated landslide probability.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Web Application

- Streamlit

### Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## 📁 Project Structure

```text
landslide-susceptibility-classification/
│
├── app.py
├── landslide_dataset.csv
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── dashboard.png
    ├── model-comparison.png
    ├── confusion-matrix.png
    └── prediction.png
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone 

Navigate into the project directory:

```bash
cd landslide-susceptibility-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The main dependencies are:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## ☁️ Deployment

This application can be deployed using **Streamlit Community Cloud**.

### Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the repository.
5. Select `app.py` as the main file.
6. Deploy the application.

Make sure `requirements.txt` is present in the repository.

---

## 📌 Example Prediction

The user provides environmental conditions such as:

```text
Rainfall          → 100 mm
Slope Angle       → 30°
Soil Saturation   → 0.50
Vegetation Cover  → 0.50
Earthquake Activity → 0.20
Proximity to Water  → 0.50
Soil Type         → Sand
```

The selected machine learning model processes these features and returns a predicted class and probability.

---

## ⚠️ Limitations

- The dataset does not contain geographical coordinates or GIS raster layers.
- The application is intended as a machine learning classification demonstration.
- The dataset may not represent real-world geological conditions.
- The probability-based Low/Moderate/High susceptibility labels used in the application are illustrative.
- The supplied 3,200-row dataset is used as the original data. SMOTE rows are
  generated only in memory during training and should not be treated as
  independent field observations.
- Real-world landslide susceptibility mapping requires validated geological, geographical, climatic, and spatial data.

---

## 🔮 Future Scope

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- Larger real-world datasets
- GIS integration
- Geographic coordinates and elevation data
- Digital Elevation Models (DEM)
- Rainfall time-series data
- Soil and geological maps
- Explainable AI techniques
- Real-time environmental data
- Interactive susceptibility maps
- Advanced models such as XGBoost and neural networks

---

## 👨‍💻 Author

**Satyam Kumar Yadav**

B.Tech Computer Science Engineering  
Specialization: Artificial Intelligence & Machine Learning

### GitHub

`https://github.com/SatyamYadav26`

---

## ⭐ Project Highlights

```text
🌋 Landslide Classification
🤖 3 Machine Learning Models
📊 Model Performance Analysis
📈 ROC Curve Analysis
🔲 Confusion Matrix Visualization
🌳 Feature Importance
🔮 Interactive Prediction
🚀 Streamlit Web Application
☁️ Deployment Ready
```

---

## 📄 License

This project is intended for educational and academic purposes.
