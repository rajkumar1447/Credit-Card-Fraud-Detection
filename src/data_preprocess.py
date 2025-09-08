# Import necessary libraries and Packages.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the dataset.
def load_data(file_path: str):
    """Load the dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df

# Function to pre process the data.
def preprocess_data(df: pd.DataFrame):
    """Split features and target, scale features, and return train/test sets."""
    x = df.drop('Class', axis=1)
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test
