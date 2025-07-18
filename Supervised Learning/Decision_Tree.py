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


print(f"Is missing value present: {is_missing_values(housing_data)}")

# Check for duplicate in the data


def is_duplicate(data):
    """Return True if duplicate exist, False otherwise."""
    return data.duplicated().values.any()


print(f"Is duplicate present: {is_duplicate(housing_data)}")
