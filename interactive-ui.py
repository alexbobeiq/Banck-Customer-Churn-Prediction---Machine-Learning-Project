import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    model = joblib.load('churn_model_xgb.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    columns = joblib.load('model_columns.pkl')

    return model, scaler, columns

model, scaler, columns = load_models()

st.title("Churn Predictor")
st.write("Please input the necessary data to determinine if the client is prone to account closing")

st.header("Info")
geography = st.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 95, 40)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0, 500000)
numOfProducts = st.slider("Number of products bought", 1, 4, 2)
is_active = st.selectbox("Is active member", ["Yes", "No"])
credit_score = st.slider("Credit score", 300, 850, 650)
estimated_salary = st.number_input("Estimated salary", min_value=0.0, value=60000.0)
has_crcard = st.selectbox("Has credit card", ["Yes", "N0"])
point_earned = st.slider("Points Earned", 0, 1000, 500)
card_type = st.selectbox("Card type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
satisfaction = st.slider("Satisfaction", 1, 5, 3)

if st.button("Make prediction", type="primary", use_container_width=True):
    inputData = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': numOfProducts,
        'HasCrCard': 1 if has_crcard == "Yes" else 0,
        'IsActiveMember': 1 if is_active == "Yes" else 0,
        'EstimatedSalary': estimated_salary,
        'Point Earned': point_earned,
        'Satisfaction Score': satisfaction,
        'Geography': geography,
        'Gender': gender,
        'Card Type': card_type
    }

    input_df = pd.DataFrame([inputData])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_df_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_df_scaled)[0][1]
    prediction = model.predict(input_df_scaled)[0]

    st.markdown("---")
    st.subheader("Results:")

    if prediction == 1:
        st.error(f"This client has the probability of {probability*100:.2f}% to leave the bank")
        st.write("Likely to leave")
    else:
        st.error(f"This client has the probability of {probability*100:.2f}% to leave the bank")
        st.write("Unlikely to leave")


