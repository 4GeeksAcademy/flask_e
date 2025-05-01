# Importación de librerías necesarias
from flask import Flask, jsonify, request
from pickle import load as pload
import logging
import numpy as np
import joblib

# Inicializar la aplicación Flask
app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Cargar el modelo y el escalador
scaler_path = 'scaler.pkl'
model_path = 'penguins_model.pkl'

with open(scaler_path, 'rb') as file:
    scaler = pload(file)

with open(model_path, 'rb') as file:
    model = pload(file)

# Ruta de prueba para verificar que Flask está funcionando
@app.route("/", methods=["GET"])
def home():
    return "¡Bienvenido a la API de clasificación de pingüinos!"

# Ruta para hacer predicciones con el modelo
@app.route("/predict", methods=["POST"])
def predict_penguin_species():
    if request.method == "POST":
        data = request.get_json()

        # Obtener los valores de entrada
        bill_length = data.get("bill_length")
        bill_depth = data.get("bill_depth")
        flipper_length = data.get("flipper_length")
        body_mass = data.get("body_mass")

        try:
            # Validación de entrada
            if None in [bill_length, bill_depth, flipper_length, body_mass]:
                logger.error("Missing value in input data")
                return jsonify({"error": "Missing value in input data"}), 400

            # Preprocesar los datos
            features = np.array([bill_length, bill_depth, flipper_length, body_mass]).reshape(1, -1)
            features_scaled = scaler.transform(features)

            # Hacer predicción
            prediction = model.predict(features_scaled)[0]
            logger.info("Prediction successful")

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": "Invalid input data"}), 400

        return jsonify({"species": prediction})

# Ejecutar la aplicación Flask en Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)