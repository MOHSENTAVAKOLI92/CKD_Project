import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_and_clean_data():
    # 1. Load the data
    print("Loading data...")
    data = pd.read_csv('data/kidney_disease.csv')
    print(f"Original data shape: {data.shape}")

    # 2. Clean column names
    data.columns = data.columns.str.strip().str.lower()

    # 3. Handle the classification column first
    data['classification'] = data['classification'].replace({'ckd\t': 'ckd', 'notckd\t': 'notckd'})

    # 4. Convert categorical values to proper format
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Create mapping dictionaries
    categorical_mappings = {
        'rbc': {'normal': 1, 'abnormal': 0},
        'pc': {'normal': 1, 'abnormal': 0},
        'pcc': {'present': 1, 'notpresent': 0},
        'ba': {'present': 1, 'notpresent': 0},
        'htn': {'yes': 1, 'no': 0},
        'dm': {'yes': 1, 'no': 0},
        'cad': {'yes': 1, 'no': 0},
        'appet': {'good': 1, 'poor': 0},
        'pe': {'yes': 1, 'no': 0},
        'ane': {'yes': 1, 'no': 0}
    }

    # Apply mappings
    for column in categorical_columns:
        data[column] = data[column].str.strip().str.lower()
        data[column] = data[column].map(categorical_mappings[column])

    # 5. Convert numerical columns to proper type
    numerical_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

    for column in numerical_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # 6. Handle missing values
    print("\nMissing values before imputation:")
    print(data.isnull().sum())

    # For numerical columns: impute with median
    for column in numerical_columns:
        data[column].fillna(data[column].median(), inplace=True)

    # For categorical columns: impute with mode
    for column in categorical_columns:
        data[column].fillna(data[column].mode()[0], inplace=True)

    # 7. Handle outliers using IQR method for numerical columns
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = np.where(data[column] > upper_bound, upper_bound,
                                np.where(data[column] < lower_bound, lower_bound, data[column]))

    # 8. Create binary target variable (1 for ckd, 0 for notckd)
    data['classification'] = data['classification'].map({'ckd': 1, 'notckd': 0})

    # 9. Scale numerical features
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Save feature columns for later use
    feature_columns = numerical_columns + categorical_columns

    # Save scaling parameters and feature columns
    scaling_params = {
        'scaler': scaler,
        'numerical_columns': numerical_columns,
        'feature_columns': feature_columns,
        'categorical_columns': categorical_columns
    }

    joblib.dump(scaling_params, 'data/scaling_params.pkl')

    # 10. Save cleaned data
    data.to_csv('data/cleaned_kidney_disease.csv', index=False)
    print("\nCleaned data shape:", data.shape)
    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    return data, feature_columns


if __name__ == "__main__":
    cleaned_data, feature_columns = load_and_clean_data()
    print("Analysis completed successfully!")