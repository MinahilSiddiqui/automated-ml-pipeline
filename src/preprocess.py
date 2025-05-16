from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    # Load the Iris dataset from scikit-learn
    iris = load_iris(as_frame=True)
    data = iris.frame.copy()
    # Add target column
    data['target'] = iris.target
    return data

def handle_missing_values(data):
    # Fill missing values with mean for numeric columns (Iris has no missing, but good practice)
    return data.fillna(data.mean())

def normalize_data(data):
    scaler = StandardScaler()
    # Select numeric feature columns, exclude 'target'
    features = data.drop(columns=['target'])
    numeric_features = features.select_dtypes(include=['float64', 'int64'])

    # Scale only feature columns
    scaled_features = scaler.fit_transform(numeric_features)

    # Replace feature columns with scaled values
    data.loc[:, numeric_features.columns] = scaled_features
    print(data.head())  # Debugging line to check the first few rows of the scaled data
    return data
