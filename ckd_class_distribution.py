import pandas as pd

# Load the data
data = pd.read_csv('data/cleaned_kidney_disease.csv')

# Display class distribution for the correct target column
print("Class Distribution:")
print(data['classification_notckd'].value_counts())
