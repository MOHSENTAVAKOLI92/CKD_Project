import pandas as pd

# Load the cleaned data
data = pd.read_csv('data/cleaned_kidney_disease.csv')

# Display column names
print("Data Columns:", data.columns)
