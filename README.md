  # CKD_Project

Mohsen Tavakoli Khabaz

Project Title: Chronic Kidney Disease Data Analysis - A Comprehensive Diagnostic Study
1. Project Overview
•	The project aimed to assist nephrologists in diagnosing and managing CKD by analyzing patient data to identify significant indicators of CKD and develop a predictive model. This project was part of an internship with MedHealth Analytics, where you were tasked with delivering insights to improve clinical practices.
2. Objectives and Deliverables
•	Objectives:
o	Identify key features associated with CKD.
o	Analyze correlations between clinical measurements.
o	Develop a CKD prediction model.
o	Provide recommendations for better diagnosis and management of CKD.
•	Deliverables:
o	Cleaned dataset.
o	EDA (Exploratory Data Analysis) report.
o	Predictive model.
o	Interactive dashboard for real-time CKD prediction.
3. Methodology and Key Steps
•	Data Exploration and Cleaning:
o	Initial Data Inspection: Loaded and inspected the dataset for missing values, outliers, and feature distributions.
o	Handling Missing Values: Columns with over 20% missing values were dropped, and remaining missing values were imputed using mean or mode based on the data type.
o	Feature Encoding and Scaling: Categorical variables were converted to numerical formats, and numerical features were scaled using StandardScaler.
o	Outliers: Outliers were handled using the IQR method, which capped values at acceptable upper and lower limits.
o	Changes Implemented: Replaced inconsistent values in the 'classification' column (e.g., ckd\t and notckd\t were corrected to ckd and notckd). This step addressed a major issue, allowing the model to recognize correct class labels.
o	Output: A cleaned dataset saved as cleaned_kidney_disease.csv.
•	Challenges and Solutions:
o	The inconsistent values in the target column initially led to misclassification, which you resolved by correcting the values in the classification column.
•	Exploratory Data Analysis (EDA):
o	Statistical Summary and Visualizations: Generated histograms, box plots, and scatter plots for key features to understand distributions.
o	Correlation Analysis: Used heatmaps and correlation matrices to explore feature relationships. Key findings indicated high blood glucose, serum creatinine, and low hemoglobin levels were strongly associated with CKD.
o	Output: A detailed EDA report highlighting important variables and insights, which informed feature selection for modeling.
•	Predictive Modeling:
o	Model Choice: A Random Forest Classifier was chosen for its robustness and interpretability.
o	Data Splitting and SMOTE: The data was split into training and testing sets, with SMOTE applied to address class imbalance.
o	Grid Search for Hyperparameter Tuning: Grid search was used to optimize hyperparameters, resulting in improved model accuracy.
o	Results: High accuracy was achieved with clear insights from confusion matrix and ROC curve evaluations.
o	Model Output: The trained model, along with feature importance data, was saved as ckd_best_model_with_features.pkl.
•	Dashboard Development:
o	Design and Implementation: Streamlit was used to create an interactive dashboard where clinicians can input patient data to predict CKD risk.
o	Changes and Final Result: Initial attempts led to incorrect predictions due to mismatches between model input features and dashboard feature names. After careful alignment of feature names, the dashboard successfully provided accurate predictions.
o	Deployment Outcome: A functional dashboard displaying CKD risk probability and important features contributing to the prediction.
4. Tools and Technologies Used
•	Libraries and Tools: Pandas, Scikit-learn, Streamlit, Joblib, Matplotlib, Seaborn.
•	Version Control: The entire project was managed through GitHub, ensuring version control and collaborative access.
5. Challenges Encountered
•	Data Cleaning Issues: Inconsistent values in the classification column initially led to misclassifications. This issue was resolved by standardizing values in the column.
•	Feature Compatibility in Dashboard: Ensuring that input features in the dashboard matched the training data required thorough checks. With precise alignment, the dashboard became effective for real-time predictions.
•	Hyperparameter Optimization: Tuning model parameters through grid search improved the model’s precision and reliability, particularly in distinguishing CKD from non-CKD cases.
6. Conclusion and Recommendations
•	This project successfully delivered an accurate CKD prediction model and an interactive dashboard, providing nephrologists with a practical tool for CKD risk assessment. Insights from the EDA and model outcomes highlight important factors for early CKD detection, such as monitoring serum creatinine, hemoglobin, and blood pressure.
7. Recommendations for Clinicians
•	Routine Monitoring: Regular monitoring of serum creatinine and hemoglobin levels is recommended for early detection.
•	Managing High-risk Factors: For patients with diabetes and hypertension, additional CKD screenings could prevent disease progression.
•	Using the Dashboard: The Streamlit dashboard serves as a real-time tool for clinicians to assess CKD risk by inputting patient-specific data.
8. Project Files and Repository
•	The full project, including data, EDA visualizations, model files, and dashboard script, is available on GitHub:
o	GitHub Repository: CKD_Project

