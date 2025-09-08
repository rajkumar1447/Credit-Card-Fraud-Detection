from data_preprocess import load_data, preprocess_data
from model_training import train_model
from evaluation import evaluate_model


def main():
    # Step 1: Load the dataset
    df = load_data("data/creditcard.csv")

    # Step 2: Preprocess the dataset
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Step 3: Train the model
    model = train_model(x_train, y_train)

    # Step 4: Evaluate the model
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
