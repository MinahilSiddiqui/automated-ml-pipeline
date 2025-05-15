import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Normalize features
    scaler = MinMaxScaler()
    features = df.drop(columns=['target'])
    df[df.columns[:-1]] = scaler.fit_transform(features)
    return df
