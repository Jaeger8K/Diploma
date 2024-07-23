"""# Define column names based on the header in the image


columns = ['ex_acc', 'duration', 'cr_history', 'purpose', 'cr_am,' 'savings', 'present_em', 'installment_rate', 'personal_status_sex',
           'debtors', 'present_residence', 'property', 'age', 'other_in', 'housing', 'ex_cr', 'job', 'n_people', 'telephone', 'foreign_worker', 'class']

# Read the file
df = pd.read_csv('Datasets/german.data', sep='\s+', names=columns, header=None)
df['personal_status_sex'] = df['personal_status_sex'].replace({'A91': 'male', 'A93': 'male', 'A94': 'male', 'A92': 'female'})
df['foreign_worker'] = df['foreign_worker'].replace({'A201': 'yes', 'A202': 'no'})
df['class'] = df['class'].replace({1: 'Good', 2: 'Bad'})

print(df)

count = ((df['personal_status_sex'] == 'male') & (df['class'] == 'Good')).sum()
print(f"Number of elements with 'male' in personal_status_sex and 'Good' in class: {count}")

count = ((df['personal_status_sex'] == 'male') & (df['class'] == 'Bad')).sum()
print(f"Number of elements with 'male' in personal_status_sex and 'Bad' in class: {count}")

count = ((df['personal_status_sex'] == 'female') & (df['class'] == 'Good')).sum()
print(f"Number of elements with 'female' in personal_status_sex and 'Good' in class: {count}")

count = ((df['personal_status_sex'] == 'female') & (df['class'] == 'Bad')).sum()
print(f"Number of elements with 'female' in personal_status_sex and 'Bad' in class: {count}")"""
import pandas as pd

df = pd.read_csv('Datasets/bank-full.csv', sep=';', quotechar='"')
print(df.columns)
df['age'] = df['age'].apply(lambda x: 'young' if x < 35 else 'old')
print(df['y'].unique())
print(df['poutcome'].unique())

print(((df['poutcome'] == 'failure') & (df['y'] == 'no')).sum())
print(((df['poutcome'] == 'failure') & (df['y'] == 'yes')).sum())
print(((df['poutcome'] == 'success') & (df['y'] == 'no')).sum())
print(((df['poutcome'] == 'success') & (df['y'] == 'yes')).sum())
print(((df['poutcome'] == 'other') & (df['y'] == 'no')).sum())
print(((df['poutcome'] == 'other') & (df['y'] == 'yes')).sum())
print(((df['poutcome'] == 'unknown') & (df['y'] == 'no')).sum())
print(((df['poutcome'] == 'unknown') & (df['y'] == 'yes')).sum())