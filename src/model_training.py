from sklearn.ensemble import RandomForestClassifier

# Function to train the model.
def train_model(x_train, y_train):
    """Train Random Forest Classifier and return the trained model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model
