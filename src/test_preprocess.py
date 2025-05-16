import pandas as pd
from preprocess import handle_missing_values, normalize_data

def test_handle_missing_values():
    data = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})
    processed_data = handle_missing_values(data)
    assert processed_data.isnull().sum().sum() == 0

def test_normalize_data():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    normalized_data = normalize_data(data)
    assert normalized_data.mean().round(2).sum() == 0
