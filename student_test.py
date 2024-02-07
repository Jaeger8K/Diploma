import pandas as pd
from scipy.stats import ttest_ind


X = pd.read_csv('normalized_features.csv')
y = pd.read_csv('y.csv')
concatenated_data = pd.concat([X, y], axis=1)

group1 = concatenated_data[concatenated_data['income'] == 1]
group1 = group1.drop(columns=['income'])

group2 = concatenated_data[concatenated_data['income'] == 0]
group2 = group2.drop(columns=['income'])

# Perform the t-test
t_statistic, p_value = ttest_ind(group1, group2)

print(concatenated_data.columns)

results_df = pd.DataFrame({'Attribute': group1.columns, 'T-statistic': t_statistic, 'P-value': p_value})

# Sort the DataFrame by the absolute T-statistic value in descending order
results_df['Abs_T-statistic'] = abs(results_df['T-statistic'])
results_df = results_df.sort_values(by='Abs_T-statistic', ascending=False)

print(results_df)

temp = results_df.index.tolist()
