import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from preprocess import load_data, handle_missing_values, normalize_data

def test_model_accuracy():
    # Load and preprocess data
    data = load_data()
    data = handle_missing_values(data)

    # Split features and target
    X = data.drop(columns=["target"])
    y = data["target"]  # Ensure this is categorical

    # Normalize features only
    X = normalize_data(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load("src/model.pkl")

    # Evaluate accuracy
    y_pred = model.predict(X_test)  # Ensure predicted labels
    assert accuracy_score(y_test, y_pred) > 0.8
