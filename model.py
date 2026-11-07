# Import required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("All libraries imported successfully!")
# Load cleaned dataset

df = pd.read_csv("data/cleaned_churn.csv")

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nTarget Column Distribution:")
print(df["Churn"].value_counts())
# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove any missing values
df = df.dropna()

# Convert categorical columns into numerical columns
# Remove customer ID because it is not useful for prediction
df = df.drop(columns=["customerID"])

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include="object").columns,
    drop_first=True,
    dtype=int
)

print("\nCategorical data converted successfully!")

print("\nUpdated Dataset:")
print(df.head())

print("\nUpdated Dataset Shape:")
print(df.shape)
# Separate features and target

X = df.drop("Churn_Yes", axis=1)

y = df["Churn_Yes"]

print("\nFeatures and Target separated successfully!")

print("Features shape:", X.shape)
print("Target shape:", y.shape)
# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData splitting completed!")

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
# Create Logistic Regression model

model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")
# Make predictions on testing data

y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save the trained model

import joblib

joblib.dump(model, "churn_model.pkl")

# Save feature column names
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel saved successfully!")