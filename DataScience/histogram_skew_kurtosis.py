import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import skew,kurtosis 
data=[12,15,14,10,8,15,14,12,10,9,12,13,14,10,9,11,13,15,14,12] data_series=pd.Series(data)
11 
data_skewness=skew(data_series) 
data_kurtosis=kurtosis(data_series) 
print(f"skewness:{data_skewness}") 
print(f"Kurtosis:{data_kurtosis}") 
plt.figure(figsize=(10,6)) 
sns.histplot(data_series,bins=10,kde=True,color='blue') 
plt.title('Histrogram of the data set') 
plt.xlabel('Value') 
plt.ylabel('Frequency') 
plt.grid(True) 
plt.show() 
