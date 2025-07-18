"""Importing modules"""
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# Data Exploration


def get_data(path):
    """Function that load the breast cancer data."""
    # Loading data
    data = pd.read_csv(path)
    return data


# Setting the path
PATH = r"C:\Users\thier\OneDrive\Desktop\ML data\breast-cancer.csv"
breast_cancer_data = get_data(PATH)

# Display first five rows
# print(breast_cancer_data.head())
# # Display shape
# print(breast_cancer_data.shape)
# # Show general information of the data
# print(breast_cancer_data.info(verbose=False))
# # Show statistical analysis of the data
# print(breast_cancer_data.describe())


# Data Cleaning:

def check_for_missing_values(data):
    """Function that return true if data has missing value, false othewise."""
    if data.isnull():
        return True
    return False


check_for_missing_values(breast_cancer_data)
