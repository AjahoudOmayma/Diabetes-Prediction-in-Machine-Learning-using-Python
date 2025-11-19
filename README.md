# Diabetes-Prediction-in-Machine-Learning-using-Python
🩺 Project Overview

This project implements a machine learning solution to predict the onset of diabetes in patients based on various diagnostic features. The primary goal is to perform comprehensive data analysis, handle potential data anomalies (like outliers and zeros), and train a robust classification model.

The model used in this analysis is the k-Nearest Neighbors (KNN) Classifier.

🚀 Getting Started

Prerequisites

To run this notebook and reproduce the results, you need a Python environment with the following libraries installed:
pip install pandas numpy matplotlib seaborn scikit-learn

Project Structure

The repository structure should look like this:

Diabetes-Prediction/
├── diabetes_prediction.ipynb    # Main analysis and modeling notebook
├── diabetes.csv                 # Dataset file
└── README.md                    # Project description (this file)

📊 Data Overview

The dataset used is the Pima Indians Diabetes Database, which contains medical diagnostic information(from Kaggle).


💻 Methodology

1.Data Loading and Inspection: Initial checks for data types, summary statistics, and null/duplicate values.

2.Exploratory Data Analysis (EDA): Visualization of feature distributions and the class imbalance of the Outcome variable. Outliers in continuous features were identified using box plots.

3.Feature Scaling: Input features were scaled/normalized before training to ensure equal contribution during the distance calculation in the KNN algorithm.

4.Model Training: A k-Nearest Neighbors (KNN) classifier was trained on the preprocessed data.

5.Evaluation: The model's performance was assessed using a Confusion Matrix and a Classification Report.

