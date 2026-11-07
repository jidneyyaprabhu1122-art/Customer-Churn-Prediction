import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Display original dataset size
print("Original Dataset Shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates()

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# Display cleaned dataset
print("\nCleaned Dataset Shape:", df.shape)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")