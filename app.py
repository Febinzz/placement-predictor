from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import numpy as np
import os

print("STEP 1 - Imports completed")

# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

print("STEP 2 - Flask app created")

# ==========================================
# LOAD MODEL + SCALER
# ==========================================

print("STEP 3 - Loading model...")
model = tf.keras.models.load_model("placement_model.keras")
print("STEP 4 - Model loaded")

print("STEP 5 - Loading scaler...")
scaler = joblib.load("scaler.pkl")
print("STEP 6 - Scaler loaded")

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

    print("STEP 7 - Prediction started")

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

    print("STEP 8 - Inputs received")

    if cgpa < 7.0:
        return render_template(
            "result.html",
            prediction="Not Eligible (CGPA below cutoff)",
            probability="0.00%"
        )

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

    print("STEP 9 - Input array created")

    student = scaler.transform(student)

    print("STEP 10 - Scaling completed")

    prediction = model.predict(student, verbose=0)

    print("STEP 11 - Prediction completed")

    predicted_class = np.argmax(prediction, axis=1)[0]

    placed_probability = prediction[0][1] * 100

    result = classes[predicted_class]

    print("STEP 12 - Returning result")

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
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )