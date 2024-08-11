import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Define the path to the dataset
file_path = '/Users/ujjwalpoudel/Documents/projects/Fatal-Collisions-Predictions/src/Killed_and_Seriously_Injured.csv'

# Load the dataset
df_accidents = pd.read_csv(file_path)

"""Data preprocessing"""

# Display the first 3 rows of the dataset
print(df_accidents.head(3))

# Summary statistics for numerical features
print('\nDescriptions')
print(df_accidents.describe())

# Display the shape of the dataset
print('\nShape')
print(df_accidents.shape)

# Display data types of each column
print('\nData types')
print(df_accidents.dtypes)

# Get and sort the list of column names
print('\nColums value soreted')
print(sorted(df_accidents.columns.values))

# Check for missing values in each column
print('\nData info')
print(df_accidents.isnull().sum())
print(df_accidents.info())

# Convert column names to a list for easier manipulation
column_names = df_accidents.columns
column_names_list = list(column_names)
print(column_names)
print(column_names_list)

# Display counts of target column values
print(df_accidents["INJURY"].value_counts())

# Create a new binary column 'is_fatal' based on the 'INJURY' column
df_accidents["is_fatal"] = np.where(df_accidents["INJURY"] == "Fatal", 1, 0)
df_accidents.drop(columns=["INJURY"], inplace=True)

# Fill missing values with the mode of each column
modes = df_accidents.mode().iloc[0]
df_accidents.fillna(modes, inplace=True)

# Verify the distribution of the 'is_fatal' column
print("\nis_fatal column value counts:")
print(df_accidents["is_fatal"].value_counts())

# Visualize the relationship between 'DISTRICT' and 'is_fatal'
sns.countplot(x='DISTRICT', hue='is_fatal', data=df_accidents)
plt.show()

# Plot histograms for numerical features
df_accidents.hist(bins=50, figsize=(20, 15))
plt.show()

# Drop columns that are not needed for the model
df_accidents.drop(columns=['INDEX_','ACCNUM','OBJECTID','INVAGE','OFFSET','INITDIR','VEHTYPE','MANOEUVER',
                           'DRIVACT', 'DRIVCOND', 'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT',
                           'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'DIVISION','ACCLASS'],
                   inplace=True)

# Separate features and target variable
y = df_accidents['is_fatal']
X = df_accidents.drop('is_fatal', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=84, stratify=y)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=['number']).columns
non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns

# Define preprocessing steps for numeric features (scaling)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define preprocessing steps for categorical features (one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, non_numeric_cols)
    ])

# Define different models for classification
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC()
}

# Create pipelines for each model with SMOTE for handling class imbalance
pipelines = {name: ImbPipeline(steps=[('preprocessor', preprocessor), ('smote', SMOTE()), ('classifier', model)])
             for name, model in models.items()}

# Define parameter grids for Grid Search for each model
param_grid_lr = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__solver': ['lbfgs', 'liblinear']
}

param_grid_dt = {
    'classifier__max_depth': [3, 5, 7, 10],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

param_grid_svc = {
    'classifier__C': [0.1, 1, 6],
    'classifier__kernel': ['rbf'],
    'classifier__gamma': ['scale'],
    'smote__k_neighbors': [3, 5]  # Added SMOTE parameter
}

# Initialize GridSearchCV for each model
grid_searches = {
    'Logistic Regression': GridSearchCV(pipelines['Logistic Regression'], param_grid_lr, cv=5, n_jobs=-1),
    'Decision Tree': GridSearchCV(pipelines['Decision Tree'], param_grid_dt, cv=5, n_jobs=-1),
    'SVM': GridSearchCV(pipelines['SVM'], param_grid_svc, cv=5, n_jobs=-1)
}

# Print feature columns used in training and testing sets
print("Training features columns:", X_train.columns)
print("Testing features columns:", X_test.columns)

# Print data types of training features
print(X_train.dtypes)

# Perform Grid Search and print best parameters and classification reports for each model
for name, grid_search in grid_searches.items():
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    print(f"Classification report for {name} after tuning:")
    print(classification_report(y_test, y_pred))

# Extract the best models from Grid Search
best_models = {name: grid_search.best_estimator_ for name, grid_search in grid_searches.items()}

# Choose the best model (Decision Tree in this case) and evaluate it
best_model = best_models['Decision Tree']
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Classification report after tuning:")
print(classification_report(y_test, y_pred))
