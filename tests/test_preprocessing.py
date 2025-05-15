import pandas as pd
from preprocessing import preprocess_data

def test_preprocess_data():
    df = pd.DataFrame({'feature1': [1, None, 3], 'target': [0, 1, 1]})
    processed_df = preprocess_data(df)
    assert processed_df.isnull().sum().sum() == 0
