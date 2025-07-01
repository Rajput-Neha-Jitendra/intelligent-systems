import pandas as pd
import numpy as np

data = pd.read_csv('student.csv')
print(data)

df_filter = data[data['percentage'] > 65]
print("\nFilter: roll no > 5")
print(df_filter)

sorted_by_name = data.sort_values(by='student name')
print("\nSorted by roll no (Ascending):")
print(sorted_by_name)

sorted_by_rno =data.sort_values(by=['roll no'], ascending=False)

7
print("\nSorted by roll no (Descending):")
print(sorted_by_rno)

