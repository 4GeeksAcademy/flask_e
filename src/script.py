from utils import db_connect
engine = db_connect()

# your code here
# Your code here
# Importación de librerías necesarias
import pandas as pd  # Para manejo de datos
import seaborn as sns  # Para cargar el dataset de pingüinos
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para escalar los datos
from sklearn.ensemble import RandomForestClassifier  # Modelo de clasificación
import joblib  # Para guardar el modelo

# Paso 1: Carga del conjunto de datos
df = sns.load_dataset("penguins")  # Cargamos el dataset de pingüinos
df.dropna(inplace=True)  # Eliminamos valores nulos

# Paso 2: Desarrollo del modelo
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]  # Variables predictoras
y = df["species"]  # Variable objetivo

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Guardado del modelo y el escalador
joblib.dump(model, "penguins_model.pkl")
joblib.dump(scaler, "scaler.pkl")
