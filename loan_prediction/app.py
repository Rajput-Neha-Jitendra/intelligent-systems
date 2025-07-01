import streamlit as st
import pickle
import numpy as np

# 🔌 Load trained model
with open('loan_prediction/loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('loan_prediction/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 📈 Load accuracy
with open("loan_prediction/model_accuracy.txt", "r") as f:
    accuracy = f.read()

# 🎨 Page Design
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.markdown("""
    <style>
       .stApp {
           background-color: #fdfdfd;
           color: #2c3e50;
           background-attachment: fixed;
           background-size: cover;
           font-family: 'Segoe UI', sans-serif;
        }

        .stButton > button {
            background-color:#1976d2;
            color: white;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color:#1565c0;  /* Slightly darker pink */
            color: white;
        }
       
.stButton > button:focus,
.stButton > button:active {
    background-color:#1565c0!important;
    color: white !important;
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}


        .title {
         border-radius: 6px;
         text-align: center;
         background-color:#1976d2;   
         color:white;                
         font-weight: bold;
         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
         padding:5px;
        }
        
        .footer {
        border-radius: 6px;
        background-color:#1976d2;  
        color:white;    
        text-align: center;
        margin-top: 50px;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# 🏦 Title and Subtitle
st.markdown('<div class="title"><h4>🏦 Loan Approval Predictor</h4><h5>Enter your details to check loan approval chances</h5></div>', unsafe_allow_html=True)

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
st.markdown("""
<div style="
    background-color:#e3f2fd;
    color:#0d47a1;
    padding: 10px 15px;
    border-radius: 6px;
    margin:10px;
    font-size: 15px;
">
 <strong>Please select correct credit history — The bank checks this and decides your loan approval request.</strong>
</div>
""", unsafe_allow_html=True)


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
        bg = "#e3f2fd"       
        color = "#0d47a1"    
    else:
        result = "❌ Sorry! Your Loan Application is Rejected"    
        bg = "#e3f2fd"       
        color = "#0d47a1"    
        
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
            border: 1px solid #ccc;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        ">
            {result}
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Made with <span style="color:red;">❤️</span> by Neha | Loan Approval Prediction using ML Logistic Regression 🚀
</div>
""", unsafe_allow_html=True)







