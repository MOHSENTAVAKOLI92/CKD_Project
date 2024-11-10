import joblib

# Load the model and feature names
model, feature_names = joblib.load('data/ckd_best_model_with_features.pkl')

# Display model feature names
print("Model Feature Names:", feature_names)
