# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset from a CSV file
data = pd.read_csv('Data science II/advertising.csv')
# Check the first few rows of the dataset to understand its structure
print(data)
# Define the independent variable (feature) and dependent variable (target)
X = data[['TV']] # Independent variable (1D array, needs to be 2D for sklearn)
y = data['Sales'] # Dependent variable (target)
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Output the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")
# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='True Values')


plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('Linear Regression Model')
plt.legend()
plt.show()
