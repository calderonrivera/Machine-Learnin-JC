import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import zscore

# =========================
# Utils
# =========================
def add_noise_to_labels(y: pd.Series, frac: float, seed: int = 42) -> pd.Series:
    if frac <= 0:
        return y
    rng = np.random.RandomState(seed)
    n_noise = int(frac * len(y))
    idx = rng.choice(y.index, size=n_noise, replace=False)
    y_noisy = y.copy()
    y_noisy.loc[idx] = 1 - y_noisy.loc[idx]
    return y_noisy

def strong_imbalance(y: pd.Series, threshold: float = 0.65) -> bool:
    # True si la clase mayoritaria supera el umbral
    p = y.mean()
    maj = max(p, 1 - p)
    return maj >= threshold

def plot_scores(scores: np.ndarray, title: str = "F1 en 50 ejecuciones"):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(1, len(scores) + 1), scores, label="F1-score")
    plt.axhline(np.mean(scores), linestyle="--", label="Media")
    plt.xlabel("Ejecución")
    plt.ylabel("F1-Score")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

def summarize_scores(scores: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"F1_Score": scores})
    df["Z_Score"] = zscore(df["F1_Score"], ddof=1)
    return df

# =========================
# Main
# =========================
def main(
    csv_path: str,
    noise_frac: float = 0.10,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 10,
):
    # 1) Cargar dataset
    df = pd.read_csv(csv_path)

    # 2) Generar etiqueta (heurística)
    #    (Se asume que las columnas existen; si no, el errors="ignore" al drop manejará ausencias)
    df["label"] = np.where(
        (df.get("tiene_dinero", 0) == 1)
        | (df.get("tiene_premio", 0) == 1)
        | ((df.get("tiene_link", 0) == 1) & (df.get("signos_exclamacion", 0) > 3)),
        1,
        0,
    ).astype(int)

    # 3) Ruido opcional
    y = df["label"]
    y = add_noise_to_labels(y, frac=noise_frac, seed=seed)

    # 4) Construir X evitando fuga + no numéricas
    leak_features = ["tiene_dinero", "tiene_premio", "tiene_link", "signos_exclamacion"]
    cols_to_drop = ["remitente", "asunto", "label"] + leak_features
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    # Verificación rápida
    if X.shape[1] == 0:
        raise ValueError(
            "No quedaron columnas numéricas en X tras eliminar fugas. "
            "Revisa el dataset o ajusta la selección de features."
        )

    # 5) Armar pipeline (imputación + árbol)
    #    Si hay desbalance fuerte, usar class_weight='balanced'
    class_weight = "balanced" if strong_imbalance(y) else None
    clf = DecisionTreeClassifier(random_state=seed, class_weight=class_weight)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    )

    # 6) Evaluación con RepeatedStratifiedKFold (≈ 50 ejecuciones)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    scorer = make_scorer(f1_score)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer, n_jobs=-1)

    # 7) Resúmenes y z-scores
    results = summarize_scores(scores)

    # 8) Entrenar en todo el set (para extra: importancias)
    pipe.fit(X, y)
    # Importancias del árbol (post-imputación las columnas son las de X)
    importancias = None
    try:
        importancias = pd.Series(
            pipe.named_steps["clf"].feature_importances_, index=X.columns
        ).sort_values(ascending=False)
    except Exception:
        pass  # Algunos clasificadores no exponen importancias

    # 9) Métricas impresas
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    n = len(scores)
    # IC 95% (aprox normal)
    half_ci = 1.96 * std / np.sqrt(n)
    ci_low, ci_high = mean - half_ci, mean + half_ci

    print("=== Métricas (RepeatedStratifiedKFold) ===")
    print(f"Media F1: {mean:.4f}")
    print(f"Desviación estándar F1: {std:.4f}")
    print(f"IC 95% F1: [{ci_low:.4f}, {ci_high:.4f}]")
    print("\nPrimeros 10 resultados:\n", results.head(10))

    # 10) Gráficas
    plot_scores(scores, title="Resultados de Árbol de Decisión (50 ejecuciones)")

    if importancias is not None:
        top_k = 10 if importancias.shape[0] >= 10 else importancias.shape[0]
        plt.figure(figsize=(10, 6))
        importancias.head(top_k).iloc[::-1].plot(kind="barh")
        plt.title("Top importancias de características (árbol)")
        plt.xlabel("Importancia")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    # 11) Exportar resultados si quieres analizarlos luego
    results.to_csv("resultados_f1.csv", index=False)
    if importancias is not None:
        importancias.to_csv("importancias_features.csv", header=["importance"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar árbol de decisión para spam/ham.")
    parser.add_argument("--csv", type=str, default="correos_features.csv", help="Ruta al CSV")
    parser.add_argument("--noise", type=float, default=0.10, help="Fracción de ruido en etiquetas [0-1]")
    parser.add_argument("--seed", type=int, default=42, help="Semilla global")
    parser.add_argument("--splits", type=int, default=5, help="Folds por repetición")
    parser.add_argument("--repeats", type=int, default=10, help="Número de repeticiones")
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        noise_frac=args.noise,
        seed=args.seed,
        n_splits=args.splits,
        n_repeats=args.repeats,
    )
