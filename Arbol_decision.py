import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from scipy.stats import zscore

# Cargar dataset
df = pd.read_csv("correos_features.csv")

# === 1. Generar etiqueta con heurística ===
df["label"] = np.where(
    (df["tiene_dinero"] == 1) | 
    (df["tiene_premio"] == 1) | 
    ((df["tiene_link"] == 1) & (df["signos_exclamacion"] > 3)),
    1, 0
)

# === 2. Introducir ruido en 10% de las etiquetas (opcional, simular datos reales) ===
rng = np.random.RandomState(42)
n_noise = int(0.10 * len(df))
noise_idx = rng.choice(df.index, size=n_noise, replace=False)
df.loc[noise_idx, "label"] = 1 - df.loc[noise_idx, "label"]

# === 3. Features (eliminar columnas de fuga y no numéricas) ===
leak_features = ["tiene_dinero", "tiene_premio", "tiene_link", "signos_exclamacion"]
X = df.drop(columns=["remitente", "asunto", "label"] + leak_features, errors="ignore")
y = df["label"]

# Asegurar solo columnas numéricas
X = X.select_dtypes(include=[np.number]).copy()

# === 4. Entrenamiento y evaluación (50 ejecuciones) ===
f1_scores = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i, stratify=y
    )
    clf = DecisionTreeClassifier(random_state=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# === 5. Resultados y z-scores ===
results = pd.DataFrame({"F1_Score": f1_scores})
results["Z_Score"] = zscore(results["F1_Score"])

# === 6. Gráfica ===
plt.figure(figsize=(10,6))
plt.scatter(range(1, 51), results["F1_Score"], label="F1-score")
plt.axhline(np.mean(results["F1_Score"]), color="red", linestyle="--", label="Media")
plt.xlabel("Ejecución")
plt.ylabel("F1-Score")
plt.title("Resultados de Árbol de Decisión en 50 ejecuciones")
plt.legend()
plt.ylim(0, 1.05)
plt.show()

# === 7. Métricas resumidas ===
print("Media F1:", round(np.mean(results["F1_Score"]), 4))
print("Desviación estándar F1:", round(np.std(results["F1_Score"]), 4))
print("\nPrimeros 10 resultados:\n", results.head(10))
