from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# ── Load the trained model ──────────────────────────────────────────────────
MODEL_PATH = 'fraud_detection_model.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅  Model loaded successfully!")
else:
    print("❌  Model file not found. Please run the training notebook first.")
    raise FileNotFoundError(
        f"'{MODEL_PATH}' does not exist. "
        "Train and save the model with joblib.dump(pipe, 'fraud_detection_model.pkl') "
        "before starting the Flask server."
    )

app = Flask(__name__)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    """Serve the web UI."""
    return render_template('index.html', model_loaded=(model is not None), model_error=None if model is not None else 'Model not loaded')


@app.route('/api', methods=['GET'])
def api_info():
    """API info for Postman/testing."""
    return jsonify({
        'message': 'Credit Card Fraud Detection API',
        'status': 'running',
        'endpoints': {
            'GET  /': 'Web UI',
            'GET  /api': 'API info',
            'POST /predict': 'Fraud detection – send JSON {"customer_name": "<name>", "card_number": "4532-XXXX-XXXX-1234", "amount": 30000}'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON body:
        { 
          "customer_name": "<name>",
          "card_number": "4532-XXXX-XXXX-1234",
          "amount": 30000
        }

    Returns:
        { 
          "result": "Legitimate" or "Fraudulent",
          "risk_level": "Low", "Medium", or "High",
          "reason": "explanation"
        }
    """
    # ── 1. Parse request ──────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({'error': 'Request Content-Type must be application/json'}), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON body'}), 400

    # ── 2. Extract and validate inputs ────────────────────────────────────
    customer_name = data.get('customer_name', '').strip()
    card_number = data.get('card_number', '').strip()
    amount = data.get('amount')

    if not customer_name or not card_number or amount is None:
        return jsonify({'error': "Missing required fields: customer_name, card_number, amount"}), 400

    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError("Amount must be positive")
    except (ValueError, TypeError) as exc:
        return jsonify({'error': f"Invalid amount: {exc}"}), 422

    # ── 3. Fraud Detection Logic (Rupees) ─────────────────────────────────
    FRAUD_THRESHOLD = 30000  # ₹30,000 threshold
    formatted_amount = int(amount) if amount.is_integer() else amount
    formatted_threshold = int(FRAUD_THRESHOLD)

    if amount > FRAUD_THRESHOLD:
        result = 'Fraud transaction'
        risk_level = 'High'
        reason = f'High-value transaction (₹{formatted_amount}) exceeds ₹{formatted_threshold} threshold'
    else:
        result = 'Normal transaction'
        risk_level = 'Low'
        reason = f'Amount (₹{formatted_amount}) is below or equal to ₹{formatted_threshold} and is a normal transaction'

    return jsonify({
        'result': result,
        'risk_level': risk_level,
        'reason': reason,
        'customer_name': customer_name,
        'card_number': card_number.replace(card_number[:-4], 'XXXX-XXXX-XXXX'),
        'amount': amount
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀  Starting Flask server …")
    print("   http://127.0.0.1:5000/\n")
    # debug=False in production; set to True only during development
    app.run(host='127.0.0.1', port=5000, debug=True)
