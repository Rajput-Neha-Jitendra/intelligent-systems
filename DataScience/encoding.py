import pandas as pd  
from sklearn.preprocessing import OneHotEncoder,LabelEncoder 
data={ 
 'Category':['A','B','A','C','B','C'], 
 'Value':[10,20,10,30,20,30] 
} 
df=pd.DataFrame(data) 
print("Original DataFrame") 
print(df) 
one_hot_encoder_df=pd.get_dummies(df,columns=['Category']) 
print(one_hot_encoder_df) 
label_encoder=LabelEncoder() 
df["Category_label"]=label_encoder.fit_transform(df['Category']) 
print(df) 
