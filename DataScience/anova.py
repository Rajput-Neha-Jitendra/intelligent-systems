import numpy as np 
import scipy.stats as stats 
import pandas as pd 
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm
# --------------- One-Way ANOVA ------------------- 
# Sample data: Replace these with your actual data 
group_A = [89, 89, 88, 78, 79] 
group_B = [93, 92, 94, 89, 88] 
group_C = [89, 88, 89, 93, 90] 
# Perform the One-Way ANOVA test 
f_statistic, p_value = stats.f_oneway(group_A, group_B, group_C) 
print("\n----- One-Way ANOVA -----") 
print(f"F-statistic: {f_statistic}") 
print(f"P-value: {p_value}") 
# Interpretation of the result 
alpha = 0.05 # significance level (5%) 
if p_value < alpha: 
 print("Reject the null hypothesis: At least one of the group means is significantly  different.") 
else: 
 print("Fail to reject the null hypothesis: The group means are not significantly  different.")

# --------------- Two-Way ANOVA ------------------- 
# Example Data: Two factors and a dependent variable 
# 'FactorA' and 'FactorB' are the two independent variables (factors) # 'Value' is the dependent variable 
data = { 
 'FactorA': ['A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2'],  'FactorB': ['B1', 'B2', 'B1', 'B2', 'B1', 'B2', 'B1', 'B2', 'B1', 'B2', 'B1', 'B2'],  'Value': [23, 21, 25, 30, 28, 31, 22, 23, 21, 35, 34, 32] 
} 
df = pd.DataFrame(data) 
print(df) 
# Perform Two-Way ANOVA 
model = ols('Value ~ C(FactorA) + C(FactorB) + C(FactorA):C(FactorB)', data=df).fit() anova_result = anova_lm(model) 
print("Two-Way ANOVA Results:") 
print(anova_result) 
# Interpretation 
# Look at the p-values to see if there is a significant effect for FactorA, FactorB, or their  interaction 
alpha = 0.05 
if anova_result['PR(>F)'][0] < alpha:
 print("FactorA has a significant effect on the dependent variable.") else: 
 print("FactorA does not have a significant effect on the dependent variable.") 
if anova_result['PR(>F)'][1] < alpha: 
 print("FactorB has a significant effect on the dependent variable.") else: 
 print("FactorB does not have a significant effect on the dependent variable.") 
if anova_result['PR(>F)'][2] < alpha: 
 print("There is a significant interaction effect between FactorA and FactorB.") else: 
 print("There is no significant interaction effect between FactorA and FactorB.") 
