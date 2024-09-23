# Import pandas for data manipulation
import pandas as pd

# Load the sample Titanic dataset from CSV file
url = r"C:\Users\ayodelet\PycharmProjects\pythonProject4\titanic_sample.csv"  # Use raw string or forward slashes
df = pd.read_csv(url)

# 1. Inspect the dataset
print("Original dataset:")
print(df)

# 2. Handle missing values
# - Fill missing 'Age' with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# - Drop the 'Cabin' column because it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

# 3. Display the cleaned dataset
print("\nCleaned dataset:")
print(df)

# 4. Save the cleaned dataset to a new CSV file
df.to_csv("C:/Users/ayodelet/PycharmProjects/pythonProject4/titanic_cleaned.csv", index=False)
