from data_loader import load_data
from train import train_model

def test_model_accuracy():
    df = load_data()
    acc = train_model(df)
    assert acc > 0.8, "Accuracy should be greater than 80%"
