import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_curve, auc
)

# ---------------------------
# 1. Cargar dataset
# ---------------------------
file_path = "correos_features.csv"
df = pd.read_csv(file_path)

# Variables
X = df.drop(columns=["remitente", "asunto", "es_largo"])  # features
y = df["es_largo"]  # target

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 2. Entrenamiento
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# 3. Evaluación
# ---------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades clase 1

# Métricas
f1 = f1_score(y_test, y_pred)
print("="*60)
print("RESULTADOS DEL MODELO")
print("="*60)
print(f"F1-Score del modelo: {f1:.3f}\n")

print("Reporte de clasificación:\n")
print(classification_report(y_test, y_pred, target_names=["HAM (0)", "SPAM (1)"]))

# ---------------------------
# 4. Gráficas
# ---------------------------

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["HAM (0)", "SPAM (1)"],
            yticklabels=["HAM (0)", "SPAM (1)"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC - Regresión Logística")
plt.legend(loc="lower right")
plt.show()

# ---------------------------
# 5. Coeficientes del modelo
# ---------------------------
print("="*60)
print("INFLUENCIA DE CADA FEATURE EN EL MODELO")
print("="*60)

feature_names = X.columns
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Crear dataframe de coeficientes
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coeficiente": coefficients,
    "Importancia": abs(coefficients),
    "Signo": ["🔺" if c > 0 else "🔻" for c in coefficients]
}).sort_values(by="Importancia", ascending=False)

# Mostrar en consola como tabla
print(coef_df.to_string(index=False, formatters={
    "Coeficiente": "{:.4f}".format,
    "Importancia": "{:.4f}".format
}))

print(f"\nIntercepto (bias): {intercept:.4f}")

# ---------------------------
# 6. Ecuación del modelo
# ---------------------------
ecuacion = "logit(p) = " + " + ".join(
    [f"({coef:.3f} * {name})" for name, coef in zip(feature_names, coefficients)]
) + f" + ({intercept:.3f})"

print("\nEcuación de la regresión logística:")
print(ecuacion)
print("\nDonde p = 1 / (1 + e^(-logit(p))) representa la probabilidad de que el correo sea SPAM (1).")

# ---------------------------
# 7. Gráfico de importancia de features
# ---------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="Coeficiente", y="Feature", data=coef_df, palette="coolwarm")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Importancia de cada Feature en la Regresión Logística")
plt.xlabel("Valor del coeficiente (impacto en logit(p))")
plt.ylabel("Feature")
plt.show()
