import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
data = { 
 'A': [1, 2, 3, 4, 5], 
 'B': [11, 25, 32, 44, 58], 
 'C': [100, 200, 300, 400, 500] 
} 
df = pd.DataFrame(data) 
print("Original Data:") 
print(df) 
min_max_scaler = MinMaxScaler()
8 
normalized_data = min_max_scaler.fit_transform(df) 
df_normalized = pd.DataFrame(normalized_data, columns=df.columns) print("\nNormalized Data (Min-Max Scaling):") 
print(df_normalized) 
standard_scaler = StandardScaler() 
standardized_data = standard_scaler.fit_transform(df) 
df_standardized = pd.DataFrame(standardized_data, columns=df.columns) print("\nStandardized Data (Z-score Scaling):") 
print(df_standardized) 
