import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn import preprocessing
dataset = 'LBW_Dataset.csv'
df = pd.read_csv(dataset)

# Perform One hot encoding for the categorical data.
y = pd.get_dummies(df['Community'], prefix = 'Community')
df = df.join(y)

# Drop redundant columns
df = df.drop(columns = 'Community')
# df = df.drop(columns = 'Education')
# df = df.drop(columns = 'Delivery phase')

# Remove outliers in the data
Q1 = df.quantile(0.05)
Q3 = df.quantile(0.95)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Imputation of NaNs as median, mode and average wherever suitable
ageMedian = int(np.median(df['Age'].dropna().values))
weightMedian = int(np.median(df['Weight'].dropna().values))
hbAverage = np.average(df['HB'].dropna().values)
bpAverage = np.average(df['BP'].dropna().values)
residenceMode = stats.mode(df['Residence'].dropna().values)[0][0]
educationMode = stats.mode(df['Education'].dropna().values)[0][0]
delphaseMode = stats.mode(df['Delivery phase'].dropna().values)[0][0]
df['Age'] = df['Age'].fillna(ageMedian).astype(int)
df['Weight'].fillna(weightMedian, inplace=True)
df['HB'].fillna(hbAverage, inplace=True)
df['BP'].fillna(bpAverage, inplace=True)
df['Residence'] = df['Residence'].fillna(residenceMode).astype(int)
df['Education'] = df['Education'].fillna(educationMode).astype(int)
df['Delivery phase'] = df['Delivery phase'].fillna(delphaseMode).astype(int)

# Normalising the DataFrame
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
# scaler = preprocessing.StandardScaler()
# x_scaled = scaler.fit_transform(df)
df = pd.DataFrame(x_scaled, columns = df.columns)
print(df)

# write the dataframe to a file
with open('../data/' + dataset[:-4] + '_clean.csv', 'w') as cleaned:
    df.to_csv(cleaned, index=False)