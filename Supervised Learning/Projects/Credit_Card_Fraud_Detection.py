
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


# Load data
credit_data = pd.read_csv(
    r"C:\Users\thier\OneDrive\Desktop\ML data\creditcard.csv")

# Display first five rows of the data.
# print(credit_data.head())

# View the shape of the data
print(credit_data.shape)

# View general info of the data
print(credit_data.info(verbose=False))

# Statistical analysis of the data
credit_data.describe()

# Check for null values
credit_data.isnull().sum()

# Check for duplicates entries
credit_data.duplicated().value_counts()

