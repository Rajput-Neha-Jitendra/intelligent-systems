import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 📥 Load dataset
df = pd.read_csv('loanp.csv')

# 🧹 Fill missing values with appropriate methods
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Education'] = df['Education'].fillna(df['Education'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# 🛠 Convert Credit_History to integer
df['Credit_History'] = df['Credit_History'].astype(int)


# Encode each column with its own encoder
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_married = LabelEncoder()
df['Married'] = le_married.fit_transform(df['Married'])

le_education = LabelEncoder()
df['Education'] = le_education.fit_transform(df['Education'])

le_self_emp = LabelEncoder()
df['Self_Employed'] = le_self_emp.fit_transform(df['Self_Employed'])

le_status = LabelEncoder()
df['Loan_Status'] = le_status.fit_transform(df['Loan_Status'])



# 🎯 Features and target
X = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📏 Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# 💾 Save model and scaler
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save all encoders
encoders = {
    'gender': le_gender,
    'married': le_married,
    'education': le_education,
    'self_emp': le_self_emp,
    'status': le_status
}

# Save model and encoders

with open('encoders.pkl', 'wb') as f:
    pickle.dump({**encoders, 'scaler': scaler}, f)

# 📈 Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 💾 Save accuracy score
with open("model_accuracy.txt", "w") as f:
    f.write(f"{round(accuracy * 100, 2)}%")

print(f"✅ Model Accuracy: {round(accuracy * 100, 2)}%")
print("🎉 Model,Accuracy, encoders, and scaler saved successfully!")
