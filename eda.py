import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_churn.csv")

# Set graph style
sns.set_style("whitegrid")

# Count customers who stayed and left
churn_counts = df["Churn"].value_counts()

# Create bar graph
plt.figure(figsize=(8, 5))

sns.barplot(
    x=churn_counts.index,
    y=churn_counts.values
)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")

# Display values on bars
for i, value in enumerate(churn_counts.values):
    plt.text(i, value + 30, str(value), ha="center")

plt.tight_layout()
plt.savefig("data/churn_distribution.png")

plt.show()
# Monthly Charges vs Customer Churn

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig("data/monthly_charges_churn.png")

plt.show()
# Contract Type vs Customer Churn

plt.figure(figsize=(9, 5))

sns.countplot(
    x="Contract",
    hue="Churn",
    data=df
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/contract_vs_churn.png")

plt.show()
# Customer Tenure vs Churn

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    multiple="stack"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/tenure_vs_churn.png")

plt.show()