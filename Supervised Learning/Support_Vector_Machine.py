"""This project utilizes two datasets sourced from Kaggle: the Breast Cancer dataset and the Digits dataset. 
For the Breast Cancer dataset, which consists of diagnostic data on tumors labeled as either benign or malignant, 
I will apply binomial logistic regression since it involves two possible outcome categories. 
The dataset includes various features extracted from digitized images of fine needle aspirates (FNA) of breast masses, such as radius, 
texture, perimeter, area, and smoothness.
In contrast, the Digits dataset contains images of handwritten digits with more than three possible categories. 
For this dataset, I will perform multinomial logistic regression to classify the digits accurately. Additionally, 
I will explore feature importance and model performance using decision trees to complement the logistic regression results.
The primary objective of this project is to develop predictive models that accurately classify breast tumors as malignant or benign, 
as well as correctly identify handwritten digits. Through exploratory data analysis and the application of machine learning techniques, 
including logistic regression.The project aims to evaluate model effectiveness and identify the most influential features for both cancer diagnosis and digit recognition.
    """

# Import required libraries.

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import seaborn as sns
# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay


# Function to get the data
def get_data(path):
    """Function that load data."""
    data = pd.read_csv(path)
    return data


PATH = r"C:\Users\thier\OneDrive\Desktop\ML data\breast-cancer.csv"
breast_cancer_data = get_data(PATH)

# Display first five rows of the data
print(f"First five rows of data: {breast_cancer_data.head()}")

# Show the shape of the data
print(f"Shape of the data: {breast_cancer_data.shape}")

# Display general info of the data
print(f"General info of the data: {breast_cancer_data.describe()}")


# Check for missing values.
def is_missing_values(data):
    """Return True if missing value exist, False otherwise."""
    if data.isnull().values.any() != 0:
        return True
    return False


# print(f"Is missing value present: {is_missing_values(housing_data)}")

# Check for duplicate in the data
def is_duplicate(data):
    """Return True if duplicate exist, False otherwise."""
    return data.duplicated().values.any()


# print(f"Is duplicate present: {is_duplicate(housing_data)}")


# """Lets check for outliers. However, since our data containt both numerical and categorical values
# we need to turn our categorical columns into numerical format. We'll be using One-hot-codding for the process."""

# Get all columns with data type object.
def get_categorical_columns(data):
    """Function that retrieve all columns with data type object."""
    return data.select_dtypes(include=['object']).columns.tolist()


# Generate dummies for columns with object as data type.
categorical_columns = get_categorical_columns(breast_cancer_data)
breast_cancer_encoded = pd.get_dummies(
    breast_cancer_data, columns=categorical_columns, drop_first=True)
# verify data type of the encoded data
print(breast_cancer_encoded.info())


# Now let proceed to the outlier detection using isolation forest method.
# For this we will create a function find_outlier.

def detect_outlier(data):
    """Function that return outlier from a dataframe if they exist."""
    # Imports the IsolationForest model from scikit-learn.
    from sklearn.ensemble import IsolationForest
    # Creates a new Isolation Forest model with a threshold of 0.05.
    # This tells the model we expect ~5% of the data to be outliers.
    model = IsolationForest(contamination=0.05)
    # Fits the Isolation Forest model to your data (excluding the target column: 'price')
    # Creates a new column called 'outlier' in the DataFrame with the prediction results.
    data['outlier'] = model.fit_predict(
        data.drop(columns='diagnosis_M', axis=1))

    return data


# Display total number of outlier
breast_cancer_with_outlier = detect_outlier(breast_cancer_encoded)
total_outlier = breast_cancer_with_outlier[breast_cancer_with_outlier['outlier'] == -1]
print(f"Total number of outlier is: {total_outlier.value_counts().sum()}")


# Visualize a comparaison between outlier data and normal data.
sns.boxplot(data=breast_cancer_with_outlier, x='outlier', y='diagnosis_M')
plt.title("Price Distribution With and Without Outliers")
plt.show()


# Further analysis of the housing dataset using decision tree model.
# We'll be modeling the housing dataset with decision tree model into two phases:  outlier and without outlier.

# 1- With outlier.

# Remove the outlier column.
breast_cancer_with_outlier = breast_cancer_with_outlier.drop(
    columns='outlier', axis=1)
# print(breast_cancer_with_outlier)


# Function that split data into training and testing datasets.
def split_data(data):
    """Separate target variable from features and Split data into training and testing datasets"""
    # Separate target variable from features
    X = data.drop(columns=['diagnosis_M'])
    y = data['diagnosis_M']
    # Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Return target, features, train and test datatests.
    return X, y, x_train, x_test, y_train, y_test


X_1, y_1, x_train_1, x_test_1, y_train_1, y_test_1 = split_data(
    breast_cancer_with_outlier)


# print(X_1.shape)
# print(y_1.shape)
# print(x_train_1.shape)
# print(x_test_1.shape)
# print(y_train_1.shape)
# print(y_test_1.shape)

# Function to scale features data
def scaler(x_train, x_test):
    """Function to scale features dataset"""
    # Initialize an instance of the StandardScaler class
    scaler = StandardScaler()
    # Scale features
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    # Return values
    return x_train_scaled, x_test_scaled


x_train_1_scaled, x_test_1_scaled = scaler(x_train_1, x_test_1)

# Train the model.


def data_training(regressor, x_train, y_train):
    """Function for dataset training"""
    # Create an instance of the SVM class
    model = regressor(kernel="rbf", gamma=0.5, C=1.0)
    # Fit the model
    model.fit(x_train, y_train)
    return model


model_1 = data_training(SVC, x_train_1_scaled, y_train_1)

# Make a prediction


def prediction(model, x_test):
    """Function that return a prediction of our model."""
    y_pred = model.predict(x_test)
    return y_pred


y_pred_1 = prediction(model_1, x_test_1_scaled)


# Perform a classification report
def report(y_test, y_pred):
    """Return a classification report of the model."""
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Classification Report", classification_report(y_test, y_pred))


# print("\nClassification report for data with outlier:")
# report(y_test_1, y_pred_1)


# REMOVING OUTLIER FROM THE HOUSE DATA AND COMPARE THE RESULT WITH PREVIOUS REPORT THAT CONTAIN OUTLIER.

breast_cancer_no_outlier = detect_outlier(breast_cancer_encoded)
breast_cancer_no_outlier = breast_cancer_no_outlier[breast_cancer_no_outlier['outlier'] == 1]

# Remove the outlier column before preprocessing.
breast_cancer_no_outlier = breast_cancer_no_outlier.drop(
    columns=['outlier'], axis=1)

print(breast_cancer_no_outlier)
# Splitting data
X_2, y_2, x_train_2, x_test_2, y_train_2, y_test_2 = split_data(
    breast_cancer_no_outlier)

# Scaling features
x_train_2_scaled, x_test_2_scaled = scaler(x_train_2, x_test_2)


# Apply PCA to reduce features to 2 components
pca = PCA(n_components=2)
x_train_2_pca = pca.fit_transform(x_train_2_scaled)
x_test_2_pca = pca.transform(x_test_2_scaled)

# Training data
model_2 = data_training(SVC, x_train_2_pca, y_train_2)

# Make prediction
y_pred_2 = prediction(model_2, x_test_2_pca)

# Report
print("\nClassification report for data without outlier:")
report(y_test_2, y_pred_2)

# Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
    model_2,
    x_train_2_pca,
    response_method="predict",
    cmap='Spectral',
    alpha=0.8,
    xlabel="PCA Component 1",
    ylabel="PCA Component 1",
)

# Scatter plot
plt.scatter(x_train_2_pca[:, 0], x_train_2_pca[:, 1],
            c=y_train_2,
            s=20, edgecolors="k")
plt.show()
