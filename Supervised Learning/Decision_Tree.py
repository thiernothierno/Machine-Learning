import pandas as pd


def get_data(path):
    """Function that load data."""
    data = pd.read_csv(path)
    return data


PATH = r"C:\Users\thier\OneDrive\Desktop\ML data\Housing.csv"
housing_data = get_data(PATH)

# Display first five rows of the data
# print(f"First five rows of data: {housing_data.head()}")

# Show the shape of the data
# print(f"Shape of the data: {housing_data.shape}")

# Display general info of the data
# print(f"General info of the data: {housing_data.describe()}")


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
categorical_columns = get_categorical_columns(housing_data)
housing_encoded = pd.get_dummies(
    housing_data, columns=categorical_columns, drop_first=True)
# verify data type of the encoded data
# print(housing_encoded.info())


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
    data['outlier'] = model.fit_predict(data.drop('price', axis=1))

    return data


# Display total number of outlier
housing_outlier = detect_outlier(housing_encoded)
total_outlier = housing_outlier[housing_outlier['outlier'] == -1]
print(f"Total number of outlier is: {total_outlier.value_counts().sum()}")
