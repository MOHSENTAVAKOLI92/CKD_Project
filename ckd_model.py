# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def train_and_evaluate_model():
    # Create necessary directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 1. Load the cleaned data
    print("Loading cleaned data...")
    data = pd.read_csv('data/cleaned_kidney_disease.csv')

    # 2. Load scaling parameters to get feature columns
    scaling_params = joblib.load('data/scaling_params.pkl')
    feature_columns = scaling_params['feature_columns']

    # 3. Prepare features and target
    X = data[feature_columns]
    y = data['classification']

    # 4. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 6. Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 7. Calculate and print metrics
    print("\nModel Performance Metrics:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 9. Save model and feature importance
    model_data = {
        'model': model,
        'feature_importance': feature_importance,
        'feature_names': feature_columns
    }
    joblib.dump(model_data, os.path.join('data', 'ckd_model.pkl'))

    print("\nModel training and evaluation completed successfully!")
    print("Model and related data saved to 'data/ckd_model.pkl'")

    return model_data


if __name__ == "__main__":
    model_data = train_and_evaluate_model()