from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_COLUMNS = [
    "Verkaufszahl", "Hubraum_L", "Kundenzufriedenheit", "Jahr", "Monat",
    "Marke_Bmw", "Marke_Mercedes-Benz", "Marke_Opel", "Marke_Volkswagen",
    "Modell_5Er", "Modell_A4", "Modell_A6", "Modell_Astra", "Modell_C-Klasse",
    "Modell_Corsa", "Modell_E-Klasse", "Modell_E-Tron", "Modell_Eqe",
    "Modell_Glc", "Modell_Golf", "Modell_Grandland", "Modell_I4",
    "Modell_Id.4", "Modell_Mokka", "Modell_Passat", "Modell_Q5",
    "Modell_Tiguan", "Modell_X5",
    "Kraftstoff_Diesel", "Kraftstoff_Elektro", "Kraftstoff_Hybrid",
    "Getriebe_Manuell",
    "Bundesland_Bayern", "Bundesland_Berlin", "Bundesland_Hamburg",
    "Bundesland_Hessen", "Bundesland_Nrw",
    "Wochentag_Monday", "Wochentag_Saturday", "Wochentag_Sunday",
    "Wochentag_Thursday", "Wochentag_Tuesday", "Wochentag_Wednesday"
]

NUMERIC_COLUMNS = [
    "Verkaufszahl", "Hubraum_L", "Kundenzufriedenheit", "Jahr", "Monat"
]


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
    data = request.get_json()

    row = {col: 0 for col in FEATURE_COLUMNS}

    row["Verkaufszahl"] = float(data.get("verkaufszahl", 0))
    row["Hubraum_L"] = float(data.get("hubraum_l", 0))
    row["Kundenzufriedenheit"] = float(data.get("kundenzufriedenheit", 0))
    row["Jahr"] = int(data.get("jahr", 2024))
    row["Monat"] = int(data.get("monat", 1))

    marke_col = "Marke_" + data.get("marke", "")
    modell_col = "Modell_" + data.get("modell", "")
    kraftstoff_col = "Kraftstoff_" + data.get("kraftstoff", "")
    bundesland_col = "Bundesland_" + data.get("bundesland", "")
    wochentag_col = "Wochentag_" + data.get("wochentag", "")

    if marke_col in row:
        row[marke_col] = 1

    if modell_col in row:
        row[modell_col] = 1

    if kraftstoff_col in row:
        row[kraftstoff_col] = 1

    if data.get("getriebe") == "Manuell":
        row["Getriebe_Manuell"] = 1

    if bundesland_col in row:
        row[bundesland_col] = 1

    if wochentag_col in row:
        row[wochentag_col] = 1

    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    input_df[NUMERIC_COLUMNS] = scaler.transform(input_df[NUMERIC_COLUMNS])

    predicted_price = model.predict(input_df)[0]

    return jsonify({
        "status": "success",
        "predicted_price_euro": round(float(predicted_price), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)