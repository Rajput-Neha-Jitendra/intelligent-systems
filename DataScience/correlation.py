import numpy as np
import scipy.stats as stats

# Function to calculate and interpret Pearson correlation
def pearson_correlation(x, y):
# Calculate Pearson correlation coefficient
  pearson_corr, p_value = stats.pearsonr(x, y)
  print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")

# Interpretation based on the value of Pearson's correlation
if pearson_corr > 0.8:
  print("Interpretation: Strong positive linear relationship")
elif pearson_corr > 0.5:
  print("Interpretation: Moderate positive linear relationship")
elif pearson_corr > 0:
  print("Interpretation: Weak positive linear relationship")
elif pearson_corr < -0.8:
  print("Interpretation: Strong negative linear relationship")
elif pearson_corr < -0.5:
  print("Interpretation: Moderate negative linear relationship")

elif pearson_corr < 0:
  print("Interpretation: Weak negative linear relationship")
else:
  print("Interpretation: No linear relationship")
  print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
  print("Conclusion: The correlation is statistically significant.")
else:
  print("Conclusion: The correlation is not statistically significant.")

# Function to calculate and interpret Spearman correlation
def spearman_correlation(x, y):
# Calculate Spearman correlation coefficient
  spearman_corr, p_value = stats.spearmanr(x, y)

print(f"Spearman Correlation Coefficient: {spearman_corr:.4f}")

# Interpretation based on the value of Spearman's correlation
if spearman_corr > 0.8:
  print("Interpretation: Strong positive monotonic relationship")
elif spearman_corr > 0.5:
  print("Interpretation: Moderate positive monotonic relationship")
elif spearman_corr > 0:
  print("Interpretation: Weak positive monotonic relationship")
elif spearman_corr < -0.8:
  print("Interpretation: Strong negative monotonic relationship")
elif spearman_corr < -0.5:
  print("Interpretation: Moderate negative monotonic relationship")
elif spearman_corr < 0:
  print("Interpretation: Weak negative monotonic relationship")
else:
  print("Interpretation: No monotonic relationship")

print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
  print("Conclusion: The correlation is statistically significant.")
else:
  print("Conclusion: The correlation is not statistically significant.")

# Example data (you can change these lists)
x = [10, 20, 30, 40, 50]
y = [12, 22, 31, 43, 48]

# Calculate and interpret Pearson correlation
print("Pearson Correlation:")
pearson_correlation(x, y)

# Calculate and interpret Spearman correlation
print("\nSpearman Correlation:")
spearman_correlation(x, y)
