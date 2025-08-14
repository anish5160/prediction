from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
import os

app = Flask(__name__)

# ----------------- TRAIN MODELS IF NOT EXIST -----------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

if not (os.path.exists("svm_model.pkl") and os.path.exists("ann_model.pkl") and os.path.exists("scaler.pkl")):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)

    ann = MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, random_state=42)
    ann.fit(X_train, y_train)

    joblib.dump(svm, "svm_model.pkl")
    joblib.dump(ann, "ann_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

# ----------------- LOAD MODELS -----------------
svm_model = joblib.load("svm_model.pkl")
ann_model = joblib.load("ann_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------- FUZZY LOGIC FUNCTION -----------------
def fuzzy_fusion_probs(svm_conf, ann_conf):
    mu_yes = max(svm_conf, ann_conf)
    mu_border = min(svm_conf, ann_conf)
    mu_no = 1 - mu_yes
    x_vals = [0.0, 0.5, 1.0]
    mu_vals = [mu_no, mu_border, mu_yes]
    numerator = sum(x * m for x, m in zip(x_vals, mu_vals))
    denominator = sum(mu_vals)
    crisp = numerator / denominator if denominator > 0 else 0.0
    return round(crisp, 2)

# ----------------- HTML (Single Page) -----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction SPA</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f6f6f6; padding: 20px; text-align: center; }
        .container { background: white; padding: 20px; border-radius: 10px; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.1);}
        input, button { padding: 8px; margin: 5px; border-radius: 5px; border: 1px solid #ccc; }
        button { background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .result { margin-top: 20px; font-size: 18px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Diabetes Prediction</h2>
        <form id="predictForm">
            <input type="number" step="any" name="pregnancies" placeholder="Pregnancies"><br>
            <input type="number" step="any" name="glucose" placeholder="Glucose"><br>
            <input type="number" step="any" name="blood_pressure" placeholder="Blood Pressure"><br>
            <input type="number" step="any" name="skin_thickness" placeholder="Skin Thickness"><br>
            <input type="number" step="any" name="insulin" placeholder="Insulin"><br>
            <input type="number" step="any" name="bmi" placeholder="BMI"><br>
            <input type="number" step="any" name="diabetes_pedigree" placeholder="Diabetes Pedigree Function"><br>
            <input type="number" step="any" name="age" placeholder="Age"><br>
            <button type="submit">Predict</button>
        </form>
        <div class="result" id="resultBox"></div>
    </div>

    <script>
        document.getElementById("predictForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            let formData = new FormData(this);
            let obj = {};
            formData.forEach((v, k) => obj[k] = v);

            let res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(obj)
            });
            let data = await res.json();

            document.getElementById("resultBox").innerHTML = `
                <p><b>SVM Probability:</b> ${data.svm_prob}</p>
                <p><b>ANN Probability:</b> ${data.ann_prob}</p>
                <p><b>Fuzzy Output:</b> ${data.fuzzy_output}</p>
                <p><b>Interpretation:</b> ${data.interpretation}</p>
            `;
        });
    </script>
</body>
</html>
"""

# ----------------- ROUTES -----------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["blood_pressure"]),
            float(data["skin_thickness"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["diabetes_pedigree"]),
            float(data["age"])
        ]
        X = scaler.transform([features])
        svm_prob = round(float(svm_model.predict_proba(X)[:, 1][0]), 3)
        ann_prob = round(float(ann_model.predict_proba(X)[:, 1][0]), 3)
        fuzzy_output = fuzzy_fusion_probs(svm_prob, ann_prob)

        if fuzzy_output >= 0.7:
            interpretation = "High chance of Diabetes"
        elif fuzzy_output >= 0.4:
            interpretation = "Borderline risk"
        else:
            interpretation = "Low risk"

        return jsonify({
            "svm_prob": svm_prob,
            "ann_prob": ann_prob,
            "fuzzy_output": fuzzy_output,
            "interpretation": interpretation
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------- RUN -----------------
if __name__ == "__main__":
    app.run()
