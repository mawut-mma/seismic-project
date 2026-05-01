from flask import Flask, request, jsonify
import joblib, pandas as pd

app   = Flask(__name__)
model = joblib.load("data/processed/seismic_model.pkl")
LABELS   = {0:"SAFE", 1:"WATCH", 2:"DANGER", 3:"EVACUATE"}
FEATURES = ["mag","depth","vibration_hz","acceleration_m2s","floor_level","fire_sensor"]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "seismic_risk_classifier"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        df   = pd.DataFrame([{f: data[f] for f in FEATURES}])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0].tolist()
        return jsonify({
            "prediction":    int(pred),
            "label":         LABELS[pred],
            "confidence":    max(prob),
            "probabilities": {LABELS[i]: round(p, 4) for i, p in enumerate(prob)}
        })
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

if __name__ == "__main__":
    print("Starting Seismic Risk API on http://localhost:5000")
    app.run(debug=True, port=5000)
