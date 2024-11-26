from flask import Flask, request, jsonify
from model_api import load_production_model

app = Flask(__name__)

# Charge le modèle de production
model = load_production_model("xgboost")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        predictions = model.predict(data["features"])
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
