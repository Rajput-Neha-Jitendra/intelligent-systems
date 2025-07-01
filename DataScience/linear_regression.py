import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
# Sample dataset for Simple and Multiple Linear Regression 
data = { 
 'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Independent variable 1 (e.g., years of experience)  'X2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # Independent variable 2 (e.g., education level)  'y': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] # Dependent variable (e.g., salary) } 
# Convert the data dictionary into a pandas DataFrame 
df = pd.DataFrame(data) 
# --- SIMPLE LINEAR REGRESSION --- 
# Let's assume 'X1' as the independent variable and 'y' as the dependent variable for  simple linear regression. 
# Prepare the data for Simple Linear Regression 
X_simple = df[['X1']] # Independent variable (e.g., years of experience) y_simple = df['y'] # Dependent variable (e.g., salary) 
# Split the data into training and testing sets

X_train_simple, X_test_simple, y_train_simple, y_test_simple =  train_test_split(X_simple, y_simple, test_size=0.2, random_state=42) 
# Create and train the model 
model_simple = LinearRegression() 
model_simple.fit(X_train_simple, y_train_simple) 
# Predict the results 
y_pred_simple = model_simple.predict(X_test_simple) 
# Calculate Mean Squared Error and R-squared 
mse_simple = mean_squared_error(y_test_simple, y_pred_simple) r2_simple = r2_score(y_test_simple, y_pred_simple) 
# Plotting the Simple Linear Regression results 
plt.figure(figsize=(8, 6)) 
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual values') plt.plot(X_test_simple, y_pred_simple, color='red', label='Regression Line') plt.xlabel('X1 (Years of Experience)') 
plt.ylabel('y (Salary)') 
plt.title('Simple Linear Regression: Salary vs Experience') plt.legend() 
plt.show()

print("\n----- Simple Linear Regression -----") 
print(f"Intercept: {model_simple.intercept_}") 
print(f"Coefficient: {model_simple.coef_}") 
print(f"Mean Squared Error: {mse_simple}") 
print(f"R-squared: {r2_simple}") 
# --- MULTIPLE LINEAR REGRESSION --- 
# Now, let's use both 'X1' (experience) and 'X2' (education level) as independent  variables. 
# Prepare the data for Multiple Linear Regression 
X_multiple = df[['X1', 'X2']] # Independent variables 
y_multiple = df['y'] # Dependent variable 
# Split the data into training and testing sets 
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple =  train_test_split(X_multiple, y_multiple, test_size=0.2, random_state=42) 
# Create and train the model 
model_multiple = LinearRegression() 
model_multiple.fit(X_train_multiple, y_train_multiple) 
# Predict the results 
y_pred_multiple = model_multiple.predict(X_test_multiple)

# Calculate Mean Squared Error and R-squared for Multiple Linear Regression mse_multiple = mean_squared_error(y_test_multiple, y_pred_multiple) r2_multiple = r2_score(y_test_multiple, y_pred_multiple) 
# Print Multiple Linear Regression results 
print("\n----- Multiple Linear Regression -----") 
print(f"Intercept: {model_multiple.intercept_}") 
print(f"Coefficients: {model_multiple.coef_}") 
print(f"Mean Squared Error: {mse_multiple}") 
print(f"R-squared: {r2_multiple}") 
# Note: Visualization for Multiple Linear Regression is complex with more than 1  independent variable, but you can plot the residuals 
# Plotting Residuals for Multiple Linear Regression 
residuals_multiple = y_test_multiple - y_pred_multiple 
plt.figure(figsize=(8, 6)) 
sns.residplot(y_pred_multiple, residuals_multiple, lowess=True, color='blue',  line_kws={'color': 'red', 'lw': 1}) 
plt.xlabel('Predicted Values') 
plt.ylabel('Residuals') 
plt.title('Residuals Plot: Multiple Linear Regression') 
plt.show() 
