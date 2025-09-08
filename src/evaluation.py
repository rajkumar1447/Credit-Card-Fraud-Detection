from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Function to evaluate the model.
def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model and print performance metrics."""
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)

    return accuracy, report, conf_matrix
