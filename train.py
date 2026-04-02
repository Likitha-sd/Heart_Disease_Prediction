import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("raw_merged_heart_dataset.csv")

# Replace ? with NaN
df.replace('?', np.nan, inplace=True)

# Columns
numeric_cols = ['age','trestbps','chol','thalachh','oldpeak']
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

# Impute
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# One-hot encoding
df = pd.get_dummies(df, columns=categorical_cols)

# Split
X = df.drop("target", axis=1)
y = df["target"]

# Scale
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train model (best from your results ✅)
model = RandomForestClassifier()
model.fit(X, y)

# Save model + columns
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
