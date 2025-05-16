from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    # Load the Iris dataset from scikit-learn
    iris = load_iris(as_frame=True)
    data = iris.frame
    # Add target column explicitly (though iris.frame already includes target)
    data['target'] = iris.target
    return data

def handle_missing_values(data):
    # Fill missing values with mean for numeric columns
    return data.fillna(data.mean())

def normalize_data(data):
    # Normalize numerical columns (expects only features, no target column)
    scaler = StandardScaler()
    numeric_features = data.select_dtypes(include=['float64', 'int64'])
    data[numeric_features.columns] = scaler.fit_transform(numeric_features)
    return data
