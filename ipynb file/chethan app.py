from flask import Flask, render_template_string, request, jsonify
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from io import BytesIO
import base64
import warnings
from sklearn.exceptions import ConvergenceWarning

# ---------------- WARNING SUPPRESSION ----------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)

app = Flask(__name__)

# ---------------- LOAD DATASET ----------------
print("Loading MNIST Dataset...")

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data[:15000] / 255.0
y = mnist.target[:15000].astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Samples:", len(x_train))
print("Testing Samples:", len(x_test))

# ---------------- MODEL ----------------
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42
)

print("Training Model...")
model.fit(x_train, y_train)

pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)

print(f"Validation Accuracy: {acc * 100:.2f}%")

# ---------------- HTML TEMPLATE ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Handwritten Digit Recognition</title>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>

        body{
            margin:0;
            padding:0;
            font-family:Arial;
            background:linear-gradient(135deg,#0f172a,#1e293b);
            color:white;
        }

        .container{
            width:95%;
            max-width:1200px;
            margin:auto;
            padding:20px;
        }

        h1{
            text-align:center;
            color:#38bdf8;
            margin-bottom:30px;
        }

        .grid{
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:25px;
        }

        .card{
            background:#1e293b;
            padding:20px;
            border-radius:15px;
            box-shadow:0 0 15px rgba(0,0,0,0.4);
        }

        canvas{
            background:white;
            border:3px solid #38bdf8;
            border-radius:10px;
            cursor:crosshair;
        }

        button{
            padding:12px 25px;
            margin:10px 5px;
            border:none;
            border-radius:10px;
            font-size:16px;
            cursor:pointer;
            font-weight:bold;
        }

        .predict-btn{
            background:#22c55e;
            color:white;
        }

        .clear-btn{
            background:#ef4444;
            color:white;
        }

        .result{
            margin-top:20px;
            font-size:22px;
            text-align:center;
        }

        .accuracy-box{
            font-size:22px;
            text-align:center;
            margin-top:20px;
            color:#facc15;
        }

        .footer{
            text-align:center;
            margin-top:20px;
            color:#94a3b8;
        }

    </style>
</head>

<body>

<div class="container">

    <h1>Handwritten Digit Recognition System</h1>

    <div class="grid">

        <!-- LEFT SIDE -->
        <div class="card">

            <h2>Draw Digit</h2>

            <canvas id="canvas" width="280" height="280"></canvas>

            <br>

            <button class="predict-btn" onclick="predictDigit()">
                Predict
            </button>

            <button class="clear-btn" onclick="clearCanvas()">
                Clear
            </button>

            <div class="result" id="result"></div>

        </div>

        <!-- RIGHT SIDE -->
        <div class="card">

            <h2>Prediction Probability Chart</h2>

            <canvas id="chartCanvas"></canvas>

            <div class="accuracy-box">
                Model Accuracy: {{accuracy}}%
            </div>

        </div>

    </div>

    <div class="footer">
        Machine Learning Based MNIST Digit Recognition using Flask
    </div>

</div>

<script>

let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0,0,280,280);

let drawing = false;

canvas.addEventListener("mousedown", start);
canvas.addEventListener("mouseup", stop);
canvas.addEventListener("mousemove", draw);

function start(e){
    drawing = true;
    ctx.beginPath();
}

function stop(e){
    drawing = false;
}

function draw(e){

    if(!drawing) return;

    let rect = canvas.getBoundingClientRect();

    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    ctx.lineTo(x,y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x,y);
}

function clearCanvas(){

    ctx.clearRect(0,0,280,280);

    ctx.fillStyle = "white";
    ctx.fillRect(0,0,280,280);

    document.getElementById("result").innerHTML = "";

    chart.data.datasets[0].data = [0,0,0,0,0,0,0,0,0,0];
    chart.update();
}

let chartCtx = document.getElementById("chartCanvas");

let chart = new Chart(chartCtx, {

    type:'bar',

    data:{
        labels:['0','1','2','3','4','5','6','7','8','9'],
        datasets:[{
            label:'Probability',
            data:[0,0,0,0,0,0,0,0,0,0]
        }]
    },

    options:{
        responsive:true,
        scales:{
            y:{
                beginAtZero:true,
                max:1
            }
        }
    }
});

function predictDigit(){

    let image = canvas.toDataURL("image/png");

    fetch("/predict",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({
            image:image
        })
    })

    .then(response => response.json())

    .then(data => {

        document.getElementById("result").innerHTML =
        "Predicted Digit : <b>" + data.prediction +
        "</b><br>Confidence : <b>" + data.confidence + "%</b>";

        chart.data.datasets[0].data = data.probabilities;
        chart.update();
    });
}

</script>

</body>
</html>
"""

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):

    image = image.convert("L")

    image = image.resize((28, 28))

    img = np.array(image)

    img = 255 - img

    img = img / 255.0

    img = img.reshape(1, 784)

    return img

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template_string(
        HTML,
        accuracy=round(acc * 100, 2)
    )

# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    image_data = data["image"]

    image_data = image_data.split(",")[1]

    image = Image.open(BytesIO(base64.b64decode(image_data)))

    processed = preprocess_image(image)

    prediction = model.predict(processed)[0]

    probabilities = model.predict_proba(processed)[0]

    confidence = round(np.max(probabilities) * 100, 2)

    return jsonify({
        "prediction": int(prediction),
        "confidence": confidence,
        "probabilities": probabilities.tolist()
    })

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False
    )