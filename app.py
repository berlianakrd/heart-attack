from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(_name_)

# ----------------------------
# Load Model & Preprocessor
# ----------------------------
model_path = os.path.join(os.path.dirname(_file_), "models", "best_model.pkl")
preprocessor_path = os.path.join(os.path.dirname(_file_), "models", "preprocessor.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input from form
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cholesterol = float(request.form["cholesterol"])
            heart_rate = float(request.form["heart_rate"])
            diabetes = float(request.form["diabetes"])
            hypertension = float(request.form["hypertension"])
            smoking = float(request.form["smoking"])
            obesity = float(request.form["obesity"])

            # Convert to array
            features = np.array([[age, sex, cholesterol, heart_rate,
                                  diabetes, hypertension, smoking, obesity]])

            # Preprocess & predict
            processed = preprocessor.transform(features)
            prediction = model.predict(processed)[0]

            result = "Tinggi" if prediction == 1 else "Rendah"

            return render_template("predict.html", prediction=result)

        except Exception as e:
            return render_template("predict.html", prediction=f"Error: {str(e)}")

    return render_template("predict.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

# ----------------------------
# Main Running Point
# ----------------------------
if _name_ == "_main_":
    app.run(debug=True)