Based on your code, this is a short and accurate **README.md**:

# 💳 Credit Card Fraud Detection System

## 📌 Overview

A Flask-based web application that detects fraudulent credit card transactions using Machine Learning. Users can enter transaction details and the system predicts whether the transaction is safe or fraudulent. The application also stores transaction history and provides analytics through a dashboard.

## 🚀 Features

* Fraud Detection using Machine Learning model
* Transaction History Management
* Interactive Dashboard & Analytics
* Fraud vs Safe Transaction Statistics
* REST API Support (JSON Requests)
* CSV-based Transaction Storage

## 🛠️ Technologies Used

* Python
* Flask
* NumPy
* Pickle
* CSV
* HTML/CSS/JavaScript
* Machine Learning

## 📊 How It Works

1. User enters transaction details.
2. The system analyzes the transaction amount and ML model prediction.
3. Transactions are classified as:

   * ✅ Safe Transaction
   * ❌ Fraudulent Transaction
4. Results are stored in a CSV file and displayed in analytics dashboards.

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── model.pkl
├── transactions.csv
├── templates/
├── static/
└── README.md
```

## ▶️ Run the Project

```bash
pip install flask numpy
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 🎯 Project Goal

To build a simple and efficient fraud detection system that helps identify suspicious credit card transactions and provides transaction monitoring through a web dashboard.

## 👨‍💻 Author

Darshu

