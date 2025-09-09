import pandas as pd
import random

# Definir número de registros
n = 1000

# Crear dataset con valores aleatorios
data = {
    "Edad": [random.randint(18, 70) for _ in range(n)],
    "Género": [random.choice(["M", "F"]) for _ in range(n)],
    "Altura_cm": [random.randint(150, 190) for _ in range(n)],
    "Peso_kg": [random.randint(50, 100) for _ in range(n)],
    "IMC": [],
    "Actividad_min": [random.randint(0, 120) for _ in range(n)],
    "Frec_Cardiaca": [random.randint(60, 100) for _ in range(n)],
    "Presion_Sistolica": [random.randint(100, 160) for _ in range(n)],
    "Nivel_Estres": [random.randint(1, 10) for _ in range(n)],
    "Trastorno_Sueño": [random.choice(["Ninguno", "Insomnio", "Apnea"]) for _ in range(n)]
}

# Calcular IMC (peso / altura^2) en metros
for peso, altura in zip(data["Peso_kg"], data["Altura_cm"]):
    imc = round(peso / ((altura/100)**2), 1)
    data["IMC"].append(imc)

# Convertir en DataFrame
df = pd.DataFrame(data)

# Guardar en CSV
df.to_csv("dataset_salud.csv", index=False)

print("Archivo CSV generado con éxito 🎉")
