import os
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import pandas as pd
import joblib
import mysql.connector
import matplotlib.pyplot as plt

# LOAD TRAINED MODEL
model = joblib.load("churn_model.pkl")

# LOAD MODEL COLUMNS
model_columns = joblib.load("model_columns.pkl")


# MYSQL DATABASE CONNECTION
def create_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database="customer_churn_db"
        )
        return connection

    except mysql.connector.Error as error:
        st.error(f"Database connection error: {error}")
        return None


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# MAIN HEADING
st.title("Customer Churn Prediction System")

st.caption(
    "An interactive machine learning dashboard that predicts "
    "whether a customer is likely to leave a telecom service."
)

st.write(
    "Welcome to the Customer Churn Prediction Dashboard. "
    "This application helps predict whether a customer "
    "is likely to leave the company."
)

st.divider()


# DASHBOARD OVERVIEW
st.subheader("Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Machine Learning Model",
        value="Logistic Regression"
    )

with col2:
    st.metric(
        label="Input Features",
        value="30"
    )

with col3:
    st.metric(
        label="Prediction Type",
        value="Customer Churn"
    )

st.divider()


# CUSTOMER INFORMATION
st.subheader("Enter Customer Details")

# PERSONAL INFORMATION
st.write("### Personal Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    tenure = st.number_input("Tenure (Months)", 0, 100, 12)

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=50.0
    )

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=600.0
    )

st.divider()


# PHONE SERVICES
st.write("### Phone Services")

col1, col2 = st.columns(2)

with col1:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing", ["No", "Yes"]
    )

st.divider()


# INTERNET SERVICES
st.write("### Internet Services")

col1, col2 = st.columns(2)

with col1:
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

st.divider()


# CONTRACT AND PAYMENT
st.write("### Contract and Payment Information")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

st.divider()


# CONVERT CUSTOMER DETAILS INTO MODEL INPUT

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=model_columns
)


# NUMERICAL INFORMATION
input_data["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges


# PERSONAL INFORMATION
input_data["gender_Male"] = 1 if gender == "Male" else 0
input_data["Partner_Yes"] = 1 if partner == "Yes" else 0
input_data["Dependents_Yes"] = 1 if dependents == "Yes" else 0


# PHONE SERVICES
input_data["PhoneService_Yes"] = 1 if phone_service == "Yes" else 0

input_data["MultipleLines_No phone service"] = (
    1 if multiple_lines == "No phone service" else 0
)

input_data["MultipleLines_Yes"] = (
    1 if multiple_lines == "Yes" else 0
)


# INTERNET SERVICE
input_data["InternetService_Fiber optic"] = (
    1 if internet_service == "Fiber optic" else 0
)

input_data["InternetService_No"] = (
    1 if internet_service == "No" else 0
)


# INTERNET-RELATED SERVICES
internet_services = {
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies
}

for service, value in internet_services.items():

    input_data[f"{service}_Yes"] = 1 if value == "Yes" else 0

    input_data[f"{service}_No internet service"] = (
        1 if value == "No internet service" else 0
    )


# CONTRACT
input_data["Contract_One year"] = (
    1 if contract == "One year" else 0
)

input_data["Contract_Two year"] = (
    1 if contract == "Two year" else 0
)


# PAPERLESS BILLING
input_data["PaperlessBilling_Yes"] = (
    1 if paperless_billing == "Yes" else 0
)


# PAYMENT METHOD
payment_columns = {
    "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
    "Electronic check": "PaymentMethod_Electronic check",
    "Mailed check": "PaymentMethod_Mailed check"
}

for payment, column in payment_columns.items():

    input_data[column] = 1 if payment_method == payment else 0


# MODEL INPUT VERIFICATION
st.subheader("Model Input Verification")

st.write("Number of input columns:", input_data.shape[1])

st.dataframe(input_data)


# CUSTOMER SUMMARY
st.divider()

st.subheader("Customer Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("Customer Tenure", f"{tenure} Months")

with summary_col2:
    st.metric("Monthly Charges", f"₹{monthly_charges:.2f}")

with summary_col3:
    st.metric("Contract Type", contract)


# CHURN PREDICTION
st.divider()
st.subheader("Churn Prediction")

if st.button("Predict Customer Churn"):

    # GENERATE PREDICTION
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # PREPARE PREDICTION VALUES
    prediction_result = "Yes" if prediction[0] == 1 else "No"

    churn_probability = probabilities[0][1] * 100
    stay_probability = probabilities[0][0] * 100

    # SAVE PREDICTION TO MYSQL
    connection = None
    cursor = None

    try:
        connection = create_connection()
        cursor = connection.cursor()

        query = """
        INSERT INTO customer_predictions
        (tenure, monthly_charges, contract, prediction, churn_probability)
        VALUES (%s, %s, %s, %s, %s)
        """

        values = (
            tenure,
            monthly_charges,
            contract,
            prediction_result,
            churn_probability
        )

        cursor.execute(query, values)
        connection.commit()

        st.success("Prediction saved successfully in MySQL!")

    except mysql.connector.Error as error:
        st.error(f"Database Error: {error}")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


    # DISPLAY PREDICTION
    if prediction[0] == 1:

        st.error("This customer is likely to leave.")

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    else:

        st.success("This customer is likely to stay.")

        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )


    # PROBABILITY METER
    st.write("### Prediction Probability")

    st.progress(int(round(churn_probability)))

    st.write(
        f"Estimated probability of leaving: {churn_probability:.2f}%"
    )


# ABOUT THE PREDICTION
st.divider()
st.subheader("About This Prediction")

st.info(
    "This prediction is generated using a Logistic Regression "
    "machine learning model trained on historical telecom "
    "customer data. The probability is an estimate, not a guarantee."
)


# FOOTER
st.divider()

st.markdown(
    "<center>Customer Churn Prediction | BSc Data Science Project By:Jidneyya Prabhu| Built with Python & Streamlit</center>",
    unsafe_allow_html=True
)


# PREDICTION HISTORY DASHBOARD
st.divider()
st.header("Customer Prediction History")

history_connection = None
history_cursor = None

try:

    # CREATE SEPARATE MYSQL CONNECTION
    history_connection = create_connection()
    history_cursor = history_connection.cursor()

    # FETCH PREVIOUS PREDICTIONS
    history_cursor.execute(
        "SELECT * FROM customer_predictions"
    )

    records = history_cursor.fetchall()

    if records:

        columns = [
            column[0] for column in history_cursor.description
        ]

        history_df = pd.DataFrame(
            records,
            columns=columns
        )

        st.subheader("All Previous Predictions")

        st.dataframe(
            history_df,
            use_container_width=True
        )


        # PREDICTION ANALYTICS
        st.divider()
        st.subheader("Prediction Analytics")

        churn_count = (
            history_df["prediction"] == "Yes"
        ).sum()

        stay_count = (
            history_df["prediction"] == "No"
        ).sum()

        total_count = len(history_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", total_count)

        with col2:
            st.metric("Likely to Leave", int(churn_count))

        with col3:
            st.metric("Likely to Stay", int(stay_count))


        # BAR CHART
        st.subheader("Churn vs Stay Analysis")

        chart_data = pd.DataFrame({
            "Customer Type": [
                "Likely to Leave",
                "Likely to Stay"
            ],
            "Number of Customers": [
                churn_count,
                stay_count
            ]
        })

        st.bar_chart(
            chart_data,
            x="Customer Type",
            y="Number of Customers"
        )


        # PIE CHART
        st.subheader("Customer Churn Distribution")

        fig, ax = plt.subplots(figsize=(4, 3))

        ax.pie(
            [churn_count, stay_count],
            labels=[
                "Likely to Leave",
                "Likely to Stay"
            ],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Customer Churn Distribution")
        ax.axis("equal")

        st.pyplot(
            fig,
            use_container_width=False
        )

        plt.close(fig)

    else:

        st.info("No prediction history available yet.")

except Exception as error:

    st.error(
        f"Error loading prediction history: {error}"
    )

finally:

    if history_cursor is not None:
        history_cursor.close()

    if history_connection is not None and history_connection.is_connected():
        history_connection.close()
