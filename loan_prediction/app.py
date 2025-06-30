import streamlit as st
import pickle
import numpy as np

# 🔌 Load trained model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 📈 Load accuracy
with open("model_accuracy.txt", "r") as f:
    accuracy = f.read()

# 🎨 Page Design
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.markdown("""
    <style>
        .title {text-align: center; color: #e91e63; font-size: 36px; font-weight: bold;}
        .subtitle {text-align: center; font-size: 18px;  margin-bottom: 30px;}
        .footer {text-align: center; margin-top: 50px; font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

# 🏦 Title and Subtitle
st.markdown('<div class="title">🏦 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your details to check loan approval chances</div>', unsafe_allow_html=True)

# 📊 Model Accuracy
st.markdown(f"<h5 style='text-align: center;'>📊 Model Info</h5><h6 style='text-align: center;'>🔍 Trained on 7 features with <span style='color:green;'>{accuracy}</span> accuracy</h6>", unsafe_allow_html=True)


# 📥 User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
credit_history = st.selectbox("Credit History", ["Yes (Good)", "No (Bad or None)"])

st.info("⚠️ Please provide actual information. Prediction is only an estimate!")

# 🔁 Encoding user input
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Yes (Good)" else 0

# 🧠 Feature vector
features = np.array([[gender, married, education, self_employed, applicant_income, loan_amount, credit_history]])

# 📏 Scale input using saved scaler
scaler = encoders['scaler']
features_scaled = scaler.transform(features)

# 🔍 Predict Button
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        result = "🎉 Congratulations! Your Loan is Approved ✅"
        bg = "#d1e7dd"
        color = "#0f5132"
    else:
        result = "❌ Sorry! Your Loan Application is Rejected"
        bg = "#f8d7da"
        color = "#842029"

    # 🖼️ Show Result
    st.markdown(f"""
        <div style="
            background-color: {bg};
            color: {color};
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        ">
            {result}
        </div>
    """, unsafe_allow_html=True)

# 👣 Footer
st.markdown('<div class="footer">Made with ❤️ by Neha | ML Loan Prediction using Logistic Regression 🚀</div>', unsafe_allow_html=True)
