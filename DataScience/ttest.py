import numpy as np
import pandas as pd
from scipy import stats

# Sample data for one-sample t-test
sample_data_one = [12, 15, 14, 10, 8, 15, 14, 12, 10, 9]
population_mean = 12 # Known population mean to test against

# Perform one-sample t-test
t_stat_one, p_value_one = stats.ttest_1samp(sample_data_one, population_mean)

# Interpretation for one-sample t-test
print("One-Sample T-Test:")
print(f"T-statistic: {t_stat_one}, P-value: {p_value_one}")
if p_value_one < 0.05:
print("Reject the null hypothesis: The sample mean is significantly different from the
population mean.")
else:
print("Fail to reject the null hypothesis: The sample mean is not significantly different
from the population mean.")

14
# Sample data for independent two-sample t-test
sample_data_a = [12, 15, 14, 10, 8]
sample_data_b = [14, 15, 13, 17, 19]

# Perform independent two-sample t-test
t_stat_two, p_value_two = stats.ttest_ind(sample_data_a, sample_data_b)

# Interpretation for independent two-sample t-test
print("\nIndependent Two-Sample T-Test:")
print(f"T-statistic: {t_stat_two}, P-value: {p_value_two}")

if p_value_two < 0.05:
print("Reject the null hypothesis: The means of the two samples are significantly
different.")
else:
print("Fail to reject the null hypothesis: The means of the two samples are not
significantly different.")
