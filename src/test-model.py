import joblib
from preprocess import load_data, handle_missing_values, normalize_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    # Load and preprocess data
    data = load_data()
    data = handle_missing_values(data)
    data = normalize_data(data)

    # Prepare features and labels
    X = data.drop(columns=["target"])
    y = data["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load("src/model.pkl")

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.8
