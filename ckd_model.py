import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned data
data = pd.read_csv('data/cleaned_kidney_disease.csv')

# Separate features and target variable
X = data.drop(columns=['classification_ckd\t'])  # Adjust as per actual target column
y = data['classification_ckd\t']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model and the feature names
joblib.dump((model, X_train.columns), 'data/ckd_best_model_with_features.pkl')

# Save a sample input template with exact feature names
X_train.iloc[:1].to_csv('data/model_input_template.csv', index=False)

print("Model training completed, saved with feature names, and input template created.")
