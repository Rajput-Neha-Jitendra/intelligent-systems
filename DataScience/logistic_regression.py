import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix 
import seaborn as sns

# Sample data (Years of experience and binary outcome: 1 means purchased, 0 means  not purchased) 
data = { 
 'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Independent variable (e.g., years of experience) 
 'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # Dependent variable (0 = No purchase, 1 =  Purchased) 
} 
# Convert the data into a pandas DataFrame 
df = pd.DataFrame(data) 
# Prepare the data for Logistic Regression 
X = df[['X']] # Independent variable (years of experience) 
y = df['y'] # Dependent variable (purchase or not) 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Create and train the Logistic Regression model 
model = LogisticRegression() 
model.fit(X_train, y_train) 
# Predict the results 
y_pred = model.predict(X_test)
# Calculate accuracy 
accuracy = accuracy_score(y_test, y_pred) 
# Confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
# Print the results 
print("\n----- Logistic Regression -----") 
print(f"Accuracy: {accuracy}") 
print(f"Confusion Matrix:\n{conf_matrix}") 
# Plotting the decision boundary for logistic regression 
plt.figure(figsize=(8, 6)) 
plt.scatter(X_train, y_train, color='blue', label='Train data') 
plt.scatter(X_test, y_test, color='red', label='Test data') 
plt.plot(X_test, model.predict_proba(X_test)[:,1], color='green', label='Decision  Boundary', linestyle='--') 
plt.xlabel('Years of Experience') 
plt.ylabel('Purchase (0 or 1)') 
plt.title('Logistic Regression: Purchase vs Experience') 
plt.legend() 
plt.show()
