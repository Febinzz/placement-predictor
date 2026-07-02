from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import numpy as np
import os

# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# LOAD MODEL + SCALER
# ==========================================

model = tf.keras.models.load_model("placement_model.keras")

scaler = joblib.load("scaler.pkl")

classes = ["Not Placed", "Placed"]

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    cgpa = float(request.form["cgpa"])
    internships = float(request.form["internships"])
    projects = float(request.form["projects"])
    workshops = float(request.form["workshops"])
    aptitude = float(request.form["aptitude"])
    softskills = float(request.form["softskills"])
    activities = float(request.form["activities"])
    training = float(request.form["training"])
    ssc = float(request.form["ssc"])
    hsc = float(request.form["hsc"])

    # ==========================================
    # HARD RULE (optional)
    # ==========================================
    if cgpa < 7.0:
        return render_template(
            "result.html",
            prediction="Not Eligible (CGPA below cutoff)",
            probability="0.00%"
        )

    # ==========================================
    # PREPARE INPUT
    # ==========================================

    student = np.array([[
        cgpa,
        internships,
        projects,
        workshops,
        aptitude,
        softskills,
        activities,
        training,
        ssc,
        hsc
    ]])

    # ==========================================
    # APPLY SCALING
    # ==========================================

    student = scaler.transform(student)

    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = model.predict(student, verbose=0)

    predicted_class = np.argmax(prediction, axis=1)[0]

    placed_probability = prediction[0][1] * 100

    result = classes[predicted_class]

    # ==========================================
    # RETURN RESULT
    # ==========================================

    return render_template(
        "result.html",
        prediction=result,
        probability=f"{placed_probability:.2f}%"
    )

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )