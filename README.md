# Credit-Card-Fraud-Detection

## 📌 Project Overview
This project implements a **Credit Card Fraud Detection** system using a Random Forest Classifier.  
It uses the popular `creditcard.csv` dataset, preprocesses the data, trains a model, and evaluates it using metrics like accuracy, classification report, and confusion matrix.
---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/rajkumar1447/Credit-Card-Fraud-Detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt

📊 Usage

Run the main script after adding the dataset:

python src/main.py

Dataset_Link: "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

📈 Output Example
lua
Copy code
Accuracy: 0.9991

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00     56864
           1       0.92      0.72      0.81        98

    accuracy                           1.00     56962
   macro avg       0.96      0.86      0.91     56962
weighted avg       1.00      1.00      1.00     56962

Confusion Matrix:
 [[56857     7]
 [   28    70]]
