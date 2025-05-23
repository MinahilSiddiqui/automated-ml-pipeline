import joblib
from preprocess import load_data, handle_missing_values, normalize_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
data = load_data()
data = handle_missing_values(data)

# Split features and target
X = data.drop(columns=["target"])
y = data["target"]

# Normalize features only
X = normalize_data(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, "src/model.pkl")
