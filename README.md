# Supply Chain Risk Prediction System

A machine learning-based web application designed to classify supply chain risk into **High Risk, Low Risk, or Moderate Risk** based on multiple operational, logistics, environmental, and supply-chain parameters.

The trained machine learning model is deployed through **Streamlit**, allowing users to enter input values and receive a predicted supply chain risk category.

## Project Overview

Supply chain operations can be affected by factors such as traffic congestion, fuel consumption, warehouse inventory, weather conditions, supplier reliability, lead time, route risk, customs clearance, driver behavior, and other operational parameters.

This project uses machine learning to analyze these factors and classify the overall supply chain risk.

## Dataset

* Total records: **32,065**
* Input features: **23**
* Classification classes:

  * High Risk
  * Low Risk
  * Moderate Risk

## Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
Multiple Classification Models
     ↓
Model Evaluation
     ↓
Decision Tree Classifier
     ↓
Model Saving
     ↓
Streamlit Deployment
```

## Models Evaluated

The following classification algorithms were evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Classifier

The Decision Tree Classifier was selected for the current deployment based on the evaluation results obtained during the project.

## Input Features

The application uses 23 features:

1. Vehicle GPS Latitude
2. Vehicle GPS Longitude
3. Fuel Consumption Rate
4. ETA Variation Hours
5. Traffic Congestion Level
6. Warehouse Inventory Level
7. Loading/Unloading Time
8. Handling Equipment Availability
9. Order Fulfillment Status
10. Weather Condition Severity
11. Port Congestion Level
12. Shipping Costs
13. Supplier Reliability Score
14. Lead Time Days
15. Historical Demand
16. IoT Temperature
17. Cargo Condition Status
18. Route Risk Level
19. Customs Clearance Time
20. Driver Behavior Score
21. Fatigue Monitoring Score
22. Disruption Likelihood Score
23. Delay Probability

## Model Performance

The evaluated classification models produced the following accuracy results:

| Model                     | Accuracy |
| ------------------------- | -------: |
| Logistic Regression       |   99.83% |
| Decision Tree Classifier  |  100.00% |
| Random Forest Classifier  |  100.00% |
| Support Vector Classifier |   98.71% |

The deployed model is the **Decision Tree Classifier**.

> Note: The unusually high classification performance should be interpreted carefully and checked for possible data leakage, duplicate records, or target-derived features before using the model in a real production environment.

## Deployment

The application is built using **Streamlit**.

The deployment structure is:

```text
Supply_Chain_Project/
│
├── app.py
├── requirements.txt
├── classification_model.pkl
├── classification_scaler.pkl
├── classification_features.pkl
└── classification_label_encoder.pkl
```

### Model Files

* `classification_model.pkl` — trained Decision Tree classification model
* `classification_scaler.pkl` — fitted StandardScaler used for preprocessing
* `classification_features.pkl` — stores the required feature names and order
* `classification_label_encoder.pkl` — converts encoded predictions into risk labels

## How the Application Works

The user enters the 23 required supply-chain parameters.

The application then performs:

```text
User Input
    ↓
Feature Ordering
    ↓
StandardScaler
    ↓
Decision Tree Classifier
    ↓
Encoded Prediction
    ↓
Label Encoder
    ↓
Risk Classification
```

The final prediction is displayed as:

* **High Risk**
* **Moderate Risk**
* **Low Risk**

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/supply-chain-risk-prediction.git
cd supply-chain-risk-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Future Improvements

* Improve dataset quality and investigate potential data leakage.
* Add more real-world supply-chain data.
* Improve categorical feature handling.
* Add probability/confidence visualization.
* Add historical prediction tracking.
* Improve the user interface and dashboard.a
* Revisit the delivery-time regression component after improving its predictive features.

B.Tech – Data Science
