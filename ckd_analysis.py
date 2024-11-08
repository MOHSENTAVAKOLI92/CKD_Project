import pandas as pd

# Load the raw data
data = pd.read_csv('data/kidney_disease.csv')

# Data cleaning steps
# Drop columns with more than 20% missing values
data = data.dropna(thresh=int(0.8 * len(data)), axis=1)

# Fill missing values with mean or mode for numerical and categorical columns
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)

# Convert categorical variables to numerical (if needed)
data = pd.get_dummies(data, drop_first=True)

# Save the cleaned data
data.to_csv('data/cleaned_kidney_disease.csv', index=False)

print("Data cleaning completed successfully and saved to 'cleaned_kidney_disease.csv'")


