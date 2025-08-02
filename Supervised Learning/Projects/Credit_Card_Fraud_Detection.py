# %%
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# %% [markdown]
# ### Data Exploration
# 1. Load data using pandas
# 2. Diplay first five rows
# 3. Show the shape of the data (rows = 284807, columns= 31)
# 4. Show the general information of the data.
# 5. Display statistical analysis.
#

# %%
# Load data
credit_data = pd.read_csv(
    r"C:\Users\thier\OneDrive\Desktop\ML data\creditcard.csv")

# Display first five rows of the data.
credit_data.head()

# %%
# View the shape of the data
credit_data.shape

# %%
# View general info of the data
credit_data.info(verbose=False)

# %%
# Statistical analysis of the data
credit_data.describe()

# %% [markdown]
# ### Data Preprocessing
# 1. Check for missing values.
# 2. Check for duplicate entries.
# 3. Display the total number of fraud vs non-fraud entries.
# 4. Found that dataset is highly imbalanced (class 0 >> class 1).
# 5. Count percentage of fraud vs non-fraud.
# 6. Check for outliers.
# 7. Handle class imbalance.
#
#
#
# #### Output:
# 1. No missing values on the dataset.
# 2. Found 1081 duplicate entries. All duplicates have been removed.
# 3. number of Fraud entries: 473 and number of Non-Fraud entries: 283253
# 4. Class 0 entries is 600 times more than Class 1 entries.
# 5. Fraud : 0.17% and Non-Fraud: 99.8%. Dataset is highly imbalanced. Only a tiny fraction of transactions are fraudulent. Accuracy is misleading: A model that always predicts "0" would be 99.83% accurate but detect no fraud.This is a common challenge in fraud detection problems.
# 6. We'll use technique SMOTE(Synthetic Minority Over-sampling Technique) + Undersampling for handling class imbalance.

# %%
# Check for null values
credit_data.isnull().sum()

# %%
# Check for duplicates entries
credit_data.duplicated().value_counts()

# %%
# Number of duplicates entries for class 0.
credit_data[(credit_data.duplicated() == True) & (
    credit_data['Class'] == 0)].value_counts().sum()

# %%
#  Number of duplicates entries for class 1.
credit_data[(credit_data.duplicated() == True) & (
    credit_data['Class'] == 1)].value_counts().sum()

# %%
# Remove duplicate entries.
credit_data = credit_data.drop_duplicates()

# %%
# Check if duplicates have been removed
credit_data.shape

# %%
# Total number of fraud vs non-fraud
print(f"0 for non-fraud and 1 for fraud: {(credit_data.Class).value_counts()}")

# %%
# Count percentage of fraud vs non-fraud
fraud = (473/283726) * 100
non_fraud = (283253/283726) * 100
print(f"fraud: {fraud}")
print(f"non-fraud: {non_fraud}")

# %%
# Check the existance of outlier in the data.
# Create an instance of the IsolationForest class.
iso_model = IsolationForest()
# Fits the Isolation Forest model to the data (excluding the target column: 'class')
# Creates a new column called 'outlier' in the DataFrame with the prediction results.
credit_data['Outlier'] = iso_model.fit_predict(
    credit_data.drop(columns="Class", axis=1))
# Display first 5 rows
credit_data.head()

# %%
# Investigate the difference between data with and without outlier.
with_outlier = credit_data[credit_data.Outlier == -1].value_counts().sum()
without_outlier = credit_data[credit_data.Outlier == 1].value_counts().sum()
print(f"Data with outlier: {with_outlier}")
print(f"Normal data: {without_outlier}")

# %%
# Creat two different dataframes: one with outlier and one without outlier.
data_with_outlier = credit_data[credit_data.Outlier == -1]
normal_data = credit_data[credit_data.Outlier == 1]

# %%
# shape of both dataframes
data_with_outlier.shape, normal_data.shape

# %%
data_with_outlier.head(2)

# %%
normal_data.head(2)

# %%


# %% [markdown]
# #### Before getting into modeling our data, let remove the outlier column from both datasets.

# %%
data_with_outlier = data_with_outlier.drop(columns='Outlier', axis=1)
normal_data = normal_data.drop(columns='Outlier', axis=1)

# %%
# Check if drop off has being successful.
print(
    f"With Outliers: {data_with_outlier.columns}, Normal data: {normal_data.columns}")

# %% [markdown]
# ### Our next step would be to model our data with both outlier and non-outlier dataset.

# %% [markdown]
# ### Data modeling
# 1. Separate features dataset from target dataset
# 2. Split data into training and testing dataset
# 3. Handle class imbalance
# 4. Fit model with different supervised learning algorithm
# 5. Evaluate model
# 6. Create a report

# %% [markdown]
# #### A brief summary on how i'm planning to handle imbalance class.
# After splitting the dataset into training and testing sets, we apply SMOTE and undersampling only to the training data to address class imbalance.  SMOTE works by creating synthetic examples of the minority class (fraud) by interpolating between existing fraud samples, generating new, realistic but slightly different fraud cases to enrich the minority class without simply duplicating records. Undersampling, on the other hand, reduces the majority class (non-fraud) by randomly removing some non-fraud samples, which prevents the model from being overwhelmed by the large number of non-fraud examples. This combination results in a more balanced training set, helping the model learn meaningful patterns from fraud cases while keeping training efficient. The test set remains untouched and imbalanced, reflecting real-world conditions, so evaluating the model on it provides an honest measure of how well it can detect fraud in practice.

# %% [markdown]
# #### To make things simple and easy to work with, I will be creating different functions including:
# 1. split_data(): This function will separate features from target and split dataset into training and testing.
# 2. handle_imbalance_data(): This will handle our imbalance data.
# 3. data_training(): This function will train the dataset using a specific model.
# 4. prediction() : Will serve to make a prediction of the model.
# 5. report(): It display a generat report.

# %%
# Function that split data into training and testing datasets.


def split_data(data):
    """Separate target variable from features and Split data into training and testing datasets"""
    # Separate target variable from features
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Return target, features, train and test datatests.
    return X, y, X_train, X_test, y_train, y_test

# Handle Imbalance data


def handle_imbalance_data(X_train, y_train):
    """Function that handle imbalance data"""
    # Oversample fraud to 5,000
    over = SMOTE(sampling_strategy={1: 100000}, random_state=42)

    # Undersample non-fraud to 25,000 → total = 30,000 (fraud: 16.6%)
    under = RandomUnderSampler(sampling_strategy={0: 100000}, random_state=42)

    pipeline = Pipeline(steps=[('o', over), ('u', under)])

    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Training the model.


def data_training(regressor, X_train, y_train):
    """Function for dataset training"""
    # Create an instance of the regressor class
    model = regressor
    # Fit the model
    model.fit(X_train, y_train)
    return model

# Make a prediction


def prediction(model, X_test):
    """Function that return a prediction of our model."""
    y_pred = model.predict(X_test)
    return y_pred


# Perform a classification report
def report(y_test, y_pred):
    """Return a classification report of the model."""
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    # print("Classification Report", classification_report(y_test, y_pred))


# %% [markdown]
# ### A- Working on normal data

# %%
# Get training and testing data.
X_normal, y_normal, X_train_normal, X_test_normal, y_train_normal, y_test_normal = split_data(
    normal_data)

# %%
# Show the shahpe of each dataset.
print(f"X: {X_normal.shape}, X_train_normal: {X_train_normal.shape} X_test_normal: {X_test_normal.shape}")
print(f"y: {y_normal.shape}, y_train_normal: {y_train_normal.shape} y_test_normal: {y_test_normal.shape}")

# %%
# balance training dataset
X_train_resampled, y_train_resampled = handle_imbalance_data(
    X_train_normal, y_train_normal)
# Display shape of resampled data
print(
    f"X_resampled: {X_train_resampled.shape}, y_resampled: {y_train_resampled.shape}")

# %%
# Training data using different classification models.
models = [RandomForestClassifier()]
# RandomForestClassifier()]
# DecisionTreeClassifier(criterion='gini',
#             max_depth=5,
#             min_samples_leaf=10,
#             random_state=42)]
# RandomForestClassifier(n_estimators=100,
#             max_depth=None,
#             max_features=10,
#             oob_score=True,
#             random_state=0),
# SVC(kernel="rbf", gamma=0.5, C=1.),
# XGBClassifier(learning_rate=0.05, n_estimators=100,
# max_depth=5, random_state=42)]


for model in models:
    new_model = data_training(model, X_train_resampled, y_train_resampled)
    y_pred_normal = prediction(new_model, X_test_normal)
    report(y_test_normal, y_pred_normal)


# %% [markdown]
#

# %%
