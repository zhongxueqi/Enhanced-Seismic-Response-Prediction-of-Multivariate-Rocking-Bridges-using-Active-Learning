import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Set global font to Times New Roman with size 24 for better readability in academic papers
# Increase DPI to 300 for high-resolution figures
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 24
plt.rcParams['figure.dpi'] = 300

# Load data from Excel file
# 'dataset.xlsx' is the input file, and 'AllFeature' is the worksheet containing all features
excel_file = pd.ExcelFile('dataset.xlsx')  # Note: Keep original filename as it's a file reference
df = excel_file.parse('AllFeature')  # Note: Keep original worksheet name as it's part of the file structure

# Extract features and target variables
# Features are the first 12 columns of the dataset
X = df.iloc[:, 0:12]
# 'Failure' is the target variable for classification task (likely indicating failure status)
y_classification = df['Failure']
# 'Pu' is the target variable for regression task (likely a performance metric)
y_regression = df['Pu']

# Create and train Random Forest Classifier
# n_estimators=100: use 100 decision trees in the ensemble
# random_state=42: ensure reproducibility of results
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y_classification)  # Train the classifier on the features and classification target

# Create and train Random Forest Regressor
# Using the same hyperparameters for consistency with the classifier
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y_regression)  # Train the regressor on the features and regression target

# Extract feature importance scores from the trained models
# feature_importances_ returns normalized importance scores (sum to 1)
importances_classification = rf_classifier.feature_importances_
importances_regression = rf_regressor.feature_importances_

# Get feature names and format specific names for better readability
# Replace 'L_B' with 'L/B' and 'B_t' with 'B/t' to represent ratios correctly
feature_names = X.columns.tolist()
feature_names = [name.replace('L_B', 'L/B').replace('B_t', 'B/t') for name in feature_names]
plt.rcParams["text.usetex"] = False  # Disable LaTeX rendering to avoid potential issues

# Store feature names and their importance scores in DataFrames for easy processing
feature_importance_classification = pd.DataFrame({
    'Feature': feature_names, 
    'Importance': importances_classification
})
feature_importance_regression = pd.DataFrame({
    'Feature': feature_names, 
    'Importance': importances_regression
})

# Sort features by importance in ascending order for horizontal bar plot (so highest is at the top)
feature_importance_classification = feature_importance_classification.sort_values(by='Importance', ascending=True)
feature_importance_regression = feature_importance_regression.sort_values(by='Importance', ascending=True)

# Merge importance scores from classification and regression tasks
# Use suffixes to distinguish between the two tasks
combined_importance = pd.merge(
    feature_importance_classification, 
    feature_importance_regression, 
    on='Feature', 
    suffixes=('_classification', '_regression')
)

# Calculate combined importance as a simple average of classification and regression importance
combined_importance['Combined_Importance'] = (
    combined_importance['Importance_classification'] + 
    combined_importance['Importance_regression']
) / 2

# Sort features by combined importance in ascending order for plotting
combined_importance = combined_importance.sort_values(by='Combined_Importance', ascending=True)

# Create a figure with 3 subplots (1 row, 3 columns) for side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Feature importance from classification task
# Use blue color scheme; all importance scores are positive so single color
colors = ['#1f77b4' for _ in feature_importance_classification['Importance']]
axes[0].barh(
    feature_importance_classification['Feature'], 
    feature_importance_classification['Importance'], 
    color=colors
)
axes[0].set_xlabel('Feature Importance')
axes[0].set_xlim(right=0.4)  # Set consistent x-axis limit for better comparison
axes[0].text(0.9, 0.02, '(a)', transform=axes[0].transAxes, 
             fontsize=24, fontweight='bold', va='bottom')  # Subplot label

# Plot 2: Feature importance from regression task
# Use green color scheme for regression results
colors = ["#398239" for _ in feature_importance_regression['Importance']]
axes[1].barh(
    feature_importance_regression['Feature'], 
    feature_importance_regression['Importance'], 
    color=colors
)
axes[1].set_xlabel('Feature Importance')
axes[1].text(0.9, 0.02, '(b)', transform=axes[1].transAxes, 
             fontsize=24, fontweight='bold', va='bottom')  # Subplot label

# Plot 3: Combined feature importance
# Use purple color scheme for combined results
colors = ['#9467bd' for _ in combined_importance['Combined_Importance']]
axes[2].barh(
    combined_importance['Feature'], 
    combined_importance['Combined_Importance'], 
    color=colors
)
axes[2].set_xlabel('Feature Importance')
axes[2].set_xlim(right=0.4)  # Match x-axis limit with first plot
axes[2].text(0.9, 0.02, '(c)', transform=axes[2].transAxes, 
             fontsize=24, fontweight='bold', va='bottom')  # Subplot label

# Adjust spacing between subplots for better readability
plt.subplots_adjust(wspace=0.1)
plt.tight_layout()  # Automatically adjust layout to prevent overlap

# Uncomment the line below to save the figure as an SVG file
# plt.savefig('./图/3.2 Feature importance.svg')  # Note: Keep original path as it's a file reference

# Display the figure
plt.show()

# Sort features by combined importance in descending order and print the results
combined_importance_sorted = combined_importance.sort_values(by='Combined_Importance', ascending=False)
print('Combined Feature Importance (Descending Order):\n', combined_importance_sorted[['Feature', 'Combined_Importance']])
