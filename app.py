from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "random_forest.pkl"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "marke": data.get("marke"),
            "modell": data.get("modell"),
            "verkaufszahl": float(data.get("verkaufszahl")),
            "kraftstoff": data.get("kraftstoff"),
            "getriebe": data.get("getriebe"),
            "hubraum_l": float(data.get("hubraum_l")),
            "bundesland": data.get("bundesland"),
            "kundenzufriedenheit": float(data.get("kundenzufriedenheit")),
            "jahr": int(data.get("jahr")),
            "monat": int(data.get("monat")),
            "wochentag": data.get("wochentag")
        }])

        predicted_price = model.predict(input_df)[0]

        return jsonify({
            "status": "success",
            "predicted_price_euro": round(float(predicted_price), 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)