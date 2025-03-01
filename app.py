
from flask import Flask, request, jsonify, render_template
from model.predict import predict_emotion

app = Flask(__name__,template_folder="template")

# ✅ Route for Home Page
@app.route("/")
def home():
    return render_template("index.html")  # HTML page for user input

# ✅ Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data from request
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        emotion = predict_emotion(text)  # Get predicted emotion
        return jsonify({"emotion": emotion})  # Return emotion as JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle errors properly

if __name__ == "__main__":
    app.run(debug=True)
