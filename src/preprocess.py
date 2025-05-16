from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    # Load the Iris dataset from scikit-learn
    iris = load_iris(as_frame=True)
    data = iris.frame
    # Add target column as the dataset has features and target separate
    data['target'] = iris.target
    return data

def handle_missing_values(data):
    # Fill missing values with mean for numeric columns
    return data.fillna(data.mean())

def normalize_data(data):
    # Normalize numerical columns
    scaler = StandardScaler()
    numeric_features = data.select_dtypes(include=['float64', 'int64'])
    data[numeric_features.columns] = scaler.fit_transform(numeric_features)
    return data
