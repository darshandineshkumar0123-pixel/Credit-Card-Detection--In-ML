from flask import Flask, render_template, request, redirect, url_for, jsonify
import csv
import os
from datetime import datetime, timedelta
import pickle
import numpy as np
import webbrowser
import threading
import time

app = Flask(__name__)

# Load the ML model
model = None
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    model = None

# CSV file for transaction history
HISTORY_FILE = 'transactions.csv'

# Initialize CSV if not exists
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Card_Number', 'Amount', 'Location', 'Date', 'Result'])

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    transactions = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                transactions.append(row)
    
    total_transactions = len(transactions)
    fraud_transactions = len([t for t in transactions if t['Result'] == 'Fraudulent Transaction'])
    safe_transactions = total_transactions - fraud_transactions
    fraud_percentage = (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0
    
    # Data for graphs
    fraud_safe_data = [fraud_transactions, safe_transactions]
    
    # Transactions per day
    from collections import defaultdict
    daily_counts = defaultdict(int)
    for t in transactions:
        daily_counts[t['Date']] += 1
    
    daily_labels = sorted(daily_counts.keys())
    daily_data = [daily_counts[date] for date in daily_labels]
    
    return render_template('index.html', 
                         total_transactions=total_transactions,
                         fraud_transactions=fraud_transactions,
                         safe_transactions=safe_transactions,
                         fraud_percentage=round(fraud_percentage, 2),
                         fraud_safe_data=fraud_safe_data,
                         daily_labels=daily_labels,
                         daily_data=daily_data)

@app.route('/detect')
def detect_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Invalid JSON body'}), 400

        customer_name = data.get('customer_name', '').strip() or data.get('person_name', 'Guest')
        card_number = data.get('card_number', '').strip()
        amount_value = data.get('amount')
        location = data.get('location', 'Online')
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

        if not card_number:
            return jsonify({'error': 'Missing card_number'}), 400

        try:
            amount = float(amount_value or 0)
        except (TypeError, ValueError):
            return jsonify({'error': 'Amount must be a number'}), 400
    else:
        customer_name = request.form.get('customer_name') or request.form.get('person_name', 'Guest')
        card_number = request.form.get('card_number', '')
        try:
            amount = float(request.form.get('amount', 0))
        except (TypeError, ValueError):
            amount = 0.0
        location = request.form.get('location', 'Online')
        date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Use ML model if available, otherwise use simple rule
    if amount > 30000:
        if model:
            # Create feature vector: Time, V1-V28, Amount
            # For demo, use 0 for V1-V28 and current time for Time
            features = [0] * 29 + [amount]  # Time + V1-V28 + Amount
            features[0] = 0  # Time
            try:
                prediction = model.predict([features])[0]
            except Exception:
                prediction = 1
            if prediction == 1:
                result = "Fraudulent Transaction"
                color = "red"
            else:
                result = "Safe Transaction"
                color = "green"
        else:
            # Fallback to simple rule
            result = "Fraudulent Transaction"
            color = "red"
    else:
        result = "Safe Transaction"
        color = "green"

    # Save to CSV
    with open(HISTORY_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([customer_name, card_number, amount, location, date, result])

    if request.is_json:
        response = {
            'result': result,
            'color': color,
            'risk_level': 'High' if result == 'Fraudulent Transaction' else 'Low',
            'reason': 'Amount exceeds threshold or model flagged it.' if result == 'Fraudulent Transaction' else 'Transaction amount is within the normal range.'
        }
        return jsonify(response)

    return render_template('index.html', prediction=result, color=color,
                           person_name=customer_name, card_number=card_number,
                           amount=amount, location=location, date=date)

@app.route('/history')
def history():
    transactions = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                transactions.append(row)
    
    # Optional filter: /history?days=30
    days_param = request.args.get('days')
    if days_param:
        try:
            days = int(days_param)
        except ValueError:
            days = None

        if days is not None and days > 0:
            cutoff = datetime.now() - timedelta(days=days)
            filtered = []
            for t in transactions:
                try:
                    transaction_date = datetime.strptime(t.get('Date', ''), '%Y-%m-%d')
                    if transaction_date >= cutoff:
                        filtered.append(t)
                except ValueError:
                    filtered.append(t)
            transactions = filtered

    # Show newest first when dates are present
    def _sort_key(t):
        try:
            return datetime.strptime(t.get('Date', ''), '%Y-%m-%d')
        except ValueError:
            return datetime.min

    transactions = sorted(transactions, key=_sort_key, reverse=True)

    return render_template('index.html', transactions=transactions)

@app.route('/analytics')
def analytics():
    transactions = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                transactions.append(row)
    
    fraud_count = 0
    fraud_percentage = 0
    fraud_pct_int = 0
    safe_pct_int = 0
    
    if transactions:
        amounts = [float(t['Amount']) for t in transactions]
        total_amount = sum(amounts)
        average_amount = total_amount / len(amounts)
        highest_transaction = max(amounts)
        fraud_count = len([t for t in transactions if t['Result'] == 'Fraudulent Transaction'])
        fraud_percentage = (fraud_count / len(transactions)) * 100
        fraud_pct_int = int(fraud_percentage)
        safe_pct_int = int(100 - fraud_percentage)
    else:
        total_amount = 0
        average_amount = 0
        highest_transaction = 0
        fraud_percentage = 0
        fraud_pct_int = 0
        safe_pct_int = 100
    
    safe_count = len(transactions) - fraud_count
    
    # Data for graphs
    fraud_safe_data = [fraud_count, safe_count]
    
    # Transactions per day
    from collections import defaultdict
    daily_counts = defaultdict(int)
    for t in transactions:
        daily_counts[t['Date']] += 1
    
    daily_labels = sorted(daily_counts.keys())
    daily_data = [daily_counts[date] for date in daily_labels]
    
    return render_template('index.html',
                         total_amount=round(total_amount, 2),
                         average_amount=round(average_amount, 2),
                         highest_transaction=round(highest_transaction, 2),
                         fraud_percentage=round(fraud_percentage, 2),
                         fraud_transactions=fraud_count,
                         safe_transactions=safe_count,
                         fraud_pct=fraud_pct_int,
                         safe_pct=safe_pct_int,
                         fraud_safe_data=fraud_safe_data,
                         daily_labels=daily_labels,
                         daily_data=daily_data)

def get_network_ip():
    """Get the machine's network IP address"""
    import socket
    try:
        # Connect to a remote server to find the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google's DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.x.x"  # Fallback

if __name__ == '__main__':
    # Get network IP
    network_ip = get_network_ip()
    
    # Print startup information
    print("\n" + "="*60)
    print("🚀 Credit Card Fraud Detection System Started!")
    print("="*60)
    print("\n📱 Access the Application:\n")
    print(f"   🖥️  Local (This Computer):")
    print(f"      http://127.0.0.1:5000")
    print(f"\n   📱 Mobile/Other Devices on Network:")
    print(f"      http://{network_ip}:5000")
    print("\n" + "="*60)
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Function to open browser after server starts
    def open_browser():
        time.sleep(2)  # Wait for server to start
        webbrowser.open('http://127.0.0.1:5000')
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', debug=True, use_reloader=False)