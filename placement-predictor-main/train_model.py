import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from model import create_model

# =======================================
# LOAD DATASET
# =======================================

df = pd.read_csv("dataset.csv")

# =======================================
# ENCODE CATEGORICAL VALUES
# =======================================

df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({
    "Yes": 1,
    "No": 0
})

df["PlacementTraining"] = df["PlacementTraining"].map({
    "Yes": 1,
    "No": 0
})

df["PlacementStatus"] = df["PlacementStatus"].map({
    "Placed": 1,
    "NotPlaced": 0
})

# =======================================
# SPLIT INPUTS AND OUTPUTS
# =======================================

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# =======================================
# FEATURE SCALING
# =======================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler for Flask app
joblib.dump(scaler, "scaler.pkl")

# =======================================
# TRAIN / TEST SPLIT
# =======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =======================================
# CREATE MODEL
# =======================================

model = create_model()

# =======================================
# COMPILE MODEL
# =======================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =======================================
# TRAIN MODEL
# =======================================

history = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# =======================================
# EVALUATE MODEL
# =======================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# =======================================
# SAVE MODEL
# =======================================

model.save("placement_model.keras")

print("\nModel trained successfully!")
print("Model saved as placement_model.keras")
print("Scaler saved as scaler.pkl")