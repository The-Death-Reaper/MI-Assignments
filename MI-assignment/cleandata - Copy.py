import pandas as pd
import scipy.stats as stats
import numpy as np

dataset = 'LBW_Dataset.csv'
df = pd.read_csv(dataset)





from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math

# Load the diabetes dataset
# diabetes_X, diabetes_y = df['Age'].dropna().values, df['Weight'].dropna().values
diabetes_X, diabetes_y = df['HB'].values, df['BP'].values
# print(diabetes_X, diabetes_y)
val_to_pop = []
for i in range(len(df['HB'].values)):
      if math.isnan(diabetes_X[i]):
            val_to_pop.append(i)
            continue
      if math.isnan(diabetes_y[i]):
            val_to_pop.append(i)
            continue
val_to_pop = set(val_to_pop)
val_to_pop = list(val_to_pop)
val_to_pop = sorted(val_to_pop, reverse=True)
# print(val_to_pop)
# np.delete()
# diabetes_X = diabetes_X[~np.isin(np.arange(diabetes_X.size), val_to_pop)]
# diabetes_y = diabetes_y[~np.isin(np.arange(diabetes_y.size), val_to_pop)]
diabetes_X = list(diabetes_X)
diabetes_y = list(diabetes_y)
for i in val_to_pop:
      diabetes_X.pop(i)
      diabetes_y.pop(i)

# from sklearn.neighbors import NearestNeighbors
# X = np.array(list(zip(diabetes_X, diabetes_y)))
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
# distances, indices = nbrs.kneighbors(np.array(df['Age'][1]).reshape(-1,1))
# print(indices, distances)

diabetes_X = np.array(diabetes_X).reshape(-1,1)
diabetes_y = np.array(diabetes_y).reshape(-1,1)
# print(list(zip(diabetes_X, diabetes_y)))
# Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20].reshape(-1,1)
# diabetes_X_test = diabetes_X[-20:].reshape(-1,1)

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20].reshape(-1,1)
# diabetes_y_test = diabetes_y[-20:].reshape(-1,1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X, diabetes_y)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X)

# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
diabetes_X_, diabetes_y_ = df['HB'].values, df['BP'].values
for i in range(len(df['HB'].values)):
      if math.isnan(diabetes_X_[i]) and not math.isnan(diabetes_y_[i]):
            # df["HB"][i] = (df['BP'][i] - regr.intercept_)/regr.coef_
            continue
      if math.isnan(diabetes_y_[i]):
            df['BP'][i] = regr.coef_*(df["HB"][i]) + regr.intercept_
            continue
# The mean squared error
# print('Mean squared error: %.2f'
      # % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
      # % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
import matplotlib.pyplot as plt

plt.scatter(diabetes_X, diabetes_y,  color='black')
plt.plot(diabetes_X, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()



# # Imputation of NaNs as median, mode and average wherever suitable
# ageMedian = int(np.median(df['Age'].dropna().values))
# weightMedian = np.average(df['Weight'].dropna().values)
# # delphaseMode = stats.mode(df['Delivery phase'].dropna().values)[0][0]
# delphaseMode = int(np.median(df['Delivery phase'].dropna().values))
# # delphaseMode = stats.mode(df['Delivery phase'].dropna().values)[0][0]
# hbAverage = np.average(df['BP'].dropna().values)
# bpAverage = np.average(df['HB'].dropna().values)
# educationMode = stats.mode(df['Education'].dropna().values)[0][0]
# residenceMode = stats.mode(df['Residence'].dropna().values)[0][0]

# df['Age'] = df['Age'].fillna(ageMedian).astype(int)
# # df['Weight'] = df['Weight'].fillna(weightMedian).astype(int)
# df['Weight'].fillna(weightMedian, inplace=True)
# df['Delivery phase'] = df['Delivery phase'].fillna(delphaseMode).astype(int)
# df['BP'].fillna(hbAverage, inplace=True)
# df['HB'].fillna(bpAverage, inplace=True)
# df['Education'] = df['Education'].fillna(educationMode).astype(int)
# df['Residence'] = df['Residence'].fillna(residenceMode).astype(int)
# # df["Result"] = 0

# with open(dataset[:-4] + '_clean.csv', 'w') as cleaned:
#     df.to_csv(cleaned, index=False)


print(df)
# print(df.max())
# print(df.min())