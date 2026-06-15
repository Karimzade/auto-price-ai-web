from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    return jsonify({
        "status": "success",
        "message": "API is working",
        "received_data": data
    }) 

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)