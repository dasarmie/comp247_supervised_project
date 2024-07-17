# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df_accidents = pd.read_csv('./src/Killed_and_Seriously_Injured.csv')
#first 3 records
print(df_accidents.head(3))
#explore numerical features
print(df_accidents.describe())
#shape of data
print(df_accidents.shape)
#explore datatypes
print(df_accidents.dtypes)
#get column list and sort
print(sorted(df_accidents.columns.values))
# Check for missing values
print(df_accidents.isnull().sum())
print(df_accidents.info())

#identify non-numeric columns
non_numeric_cols = df_accidents.select_dtypes(exclude=['number']).columns
print(f"Non-numeric columns:\t {non_numeric_cols}")

#target column
print(df_accidents["INJURY"].value_counts())
#drop rows containing null values
df_cleaned = df_accidents.dropna(subset=['INJURY'])

#define a new binary class based on column INJURY
df_accidents["is_fatal"] = df_accidents["INJURY"].apply(lambda x: 1 if x == "Fatal" else 0)

# Fill with mode (most frequent value)
df_accidents.fillna(df_accidents.mode().iloc[0], inplace=True)

# Visualize relationships
sns.countplot(x='DISTRICT', hue='is_fatal', data=df_accidents)
plt.show()
#plot histogram
df_accidents.hist(bins=50, figsize=(20, 15))
plt.show()

#identify and drop non-essential column
df_accidents.drop(columns=['INDEX_','ACCNUM','OBJECTID'], inplace=True)
#identify non-numeric columns
non_numeric_cols = df_accidents.select_dtypes(exclude=['number']).columns
#one hot encoding
df_accidents = pd.get_dummies(df_accidents, columns=non_numeric_cols)

#define X and y
y = df_accidents['is_fatal']
X = df_accidents.drop('is_fatal', axis=1)
#check distribution in target class
print(y.value_counts())
#perform oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y) # X, y are your feature and target arrays

#check distribution again
print(y_resampled.value_counts())
