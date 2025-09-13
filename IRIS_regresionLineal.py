from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. Cargar dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names  # nombres de las especies

# 2. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenar modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predicciones
y_pred = model.predict(X_test)

# Convertir predicciones continuas a clases (0, 1, 2)
y_pred_class = np.round(y_pred).astype(int)
y_pred_class = np.clip(y_pred_class, 0, 2)

# 5. Evaluación
accuracy = accuracy_score(y_test, y_pred_class)
print(f"\nPrecisión del modelo: {accuracy:.2f}\n")

# 6. Crear tabla con ejemplos del dataset y predicciones
num_muestras = 10  # Número de filas de ejemplo a mostrar
ejemplos = pd.DataFrame(X_test[:num_muestras], columns=iris.feature_names)
ejemplos["Clase real (número)"] = y_test[:num_muestras]
ejemplos["Clase real (nombre)"] = [target_names[i] for i in y_test[:num_muestras]]
ejemplos["Predicción (número)"] = y_pred_class[:num_muestras]
ejemplos["Predicción (nombre)"] = [target_names[i] for i in y_pred_class[:num_muestras]]

print("📊 Ejemplos con clases reales y predicciones:\n")
print(ejemplos.to_string(index=False))


