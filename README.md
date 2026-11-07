# Customer Churn Prediction Dashboard

## Project Overview
This project predicts whether a telecom customer is likely to leave a service using Machine Learning. It uses Logistic Regression to generate predictions and Streamlit to display an interactive dashboard.

## Features
- Customer churn prediction (Yes/No)
- Churn probability meter
- Interactive Streamlit dashboard
- MySQL database integration
- Customer prediction history
- Prediction analytics
- Bar chart and pie chart visualizations

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- MySQL
- Joblib

## Machine Learning Model
The project uses Logistic Regression for binary classification.

The model was trained using a telecom customer churn dataset.

**Model Accuracy:** Approximately 79%

## Project Files
- `app.py` – Streamlit dashboard application
- `churn_model.pkl` – Trained machine learning model
- `model_columns.pkl` – Model input columns
- `data/` – Dataset files

## How to Run the Project

1. Install Python and MySQL.
2. Install the required Python libraries.
3. Configure your MySQL database connection.
4. Open the project folder in VS Code.
5. Activate your virtual environment.
6. Run the application:

```bash
streamlit run app.py
```

## Database
Database name: `customer_churn_db`

Table name: `customer_predictions`

## Project Objective
The objective is to demonstrate how machine learning, Python, data visualization and database integration can be combined to build a practical customer churn prediction application.

## Author
BSc Data Science Student By: Jidneyya Prabhu