
# 📊 Loan Prediction App using Machine Learning
A simple and user-friendly Streamlit web app that predicts whether a loan will be approved or not based on user inputs like income, credit history, education, and more.

## 🚀 Live App
[Click here to open the Loan Prediction App](https://smartloan-predictor.streamlit.app/)

## 🧠 Project Overview
This project uses a trained **Logistic Regression model** to classify whether a loan application will be approved. The backend is built in Python using `scikit-learn`, and the frontend is created using `Streamlit`.
---

## 📂 Folder Structure
```
loan_prediction/
├──  Loan_Training.py     # Model Traning code
├── app.py                # Streamlit GUI code
├── model_accuracy.txt    # Accuracy 
├── loan_model.pkl        # Trained ML model
├── encoders.pkl          # Preprocessing encoders (LabelEncoder, OneHotEncoder, etc.)<
├── requirements.txt      # Required packages
└── README.md             # Project documentation (this file)
```
## 💡 Features

- Easy-to-use web interface
- Real-time loan approval prediction
- Trained ML model for binary classification
- Interactive inputs with dropdowns, sliders, and number fields

---

## 🛠️ Tech Stack
- Python 🐍
- Streamlit 📺
- Scikit-learn ⚙️
- Pandas 📊
- NumPy ➗

---

## 🔧 How to Run Locally

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Rajput-Neha-Jitendra/intelligent-systems.git
   cd intelligent-systems/loan_prediction
   ```

2. **Install requirements**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**  
   ```bash
   streamlit run app.py
   ```
   
## ✅ Demo Screenshot
![Loan Prediction App Screenshot](https://github.com/Rajput-Neha-Jitendra/intelligent-systems/blob/main/loan_prediction/screencapture-smartloan-predictor-streamlit-app.png)  
---

## 👤 Author
**Neha Jitendra Rajput**  
🎓 MCA Student, IMRD Shirpur
