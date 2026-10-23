"""
Lee data/processed/instances.csv (520 filas = 104 partidos x 5 checkpoints)
y evalúa Regresión Logística, SVM (RBF) y KNN con tres conjuntos de variables:

    base      : solo marcador parcial (+ minuto)
    +juego    : base + xG, tiros y tiros al arco (diferencias A - B)
    +contexto : +juego + Elo, ranking, valor del plantel, localía, descanso, fase

Validación: StratifiedGroupKFold agrupando por partido (los 5 checkpoints de un
mismo partido nunca quedan repartidos entre train y test), repetida con varias
semillas. Las métricas se calculan sobre las predicciones fuera de muestra y se
reportan por checkpoint. Los intervalos de confianza son bootstrap por partido.

Uso (desde la raíz del repo):
    python scripts/04_modelos.py
    python scripts/04_modelos.py --datos data/processed/instances.csv --repeticiones 5
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

CLASES = [0, 1, 2]                 # 0 = gana A, 1 = empate, 2 = gana B
CHECKPOINTS = [15, 30, 45, 60, 75]

BASE = ["goal_diff", "gd_x_t", "t"]
JUEGO = ["xg_diff", "tiros_diff", "al_arco_diff"]
CONTEXTO = ["elo_diff", "rank_diff", "value_diff", "host_diff", "rest_diff", "is_knockout"]
CONJUNTOS = {
    "base": BASE,
    "+juego": BASE + JUEGO,
    "+contexto": BASE + JUEGO + CONTEXTO,
}


def modelos():
    """Modelo + grilla de hiperparámetros (se elige con CV interna agrupada)."""
    lr = Pipeline([("sc", StandardScaler()),
                   ("m", LogisticRegression(max_iter=2000))])
    svm = Pipeline([("sc", StandardScaler()),
                    ("m", SVC(kernel="rbf", probability=True, random_state=0))])
    knn = Pipeline([("sc", StandardScaler()),
                    ("m", KNeighborsClassifier(weights="distance"))])
    return {
        "RegLog": (lr, {"m__C": [0.01, 0.1, 1, 10]}),
        "SVM": (svm, {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.1]}),
        "KNN": (knn, {"m__n_neighbors": [5, 15, 31, 51]}),
    }


def suavizar(P, a=0.02):
    """Suavizado mínimo para evitar log-loss infinito (sobre todo en KNN)."""
    return (1 - a) * P + a / len(CLASES)


def completar(P, clases):
    """Si en un fold falta una clase, rellena su columna con 0."""
    out = np.zeros((P.shape[0], len(CLASES)))
    for j, c in enumerate(clases):
        out[:, CLASES.index(c)] = P[:, j]
    return out


def evaluar(df, columnas, modelo, grilla, semilla, k=5):
    """Predicciones fuera de muestra con CV externa agrupada por partido."""
    X, y, g = df[columnas].values, df["y"].values, df["fotmob_id"].values
    P = np.zeros((len(df), len(CLASES)))
    externo = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
    # estratificamos por la etiqueta del partido (y es constante dentro de un partido)
    for tr, te in externo.split(X, y, g):
        gs = GridSearchCV(modelo, grilla, cv=GroupKFold(n_splits=4),
                          scoring="neg_log_loss", n_jobs=-1)
        gs.fit(X[tr], y[tr], groups=g[tr])
        P[te] = completar(gs.predict_proba(X[te]), gs.classes_)
    return suavizar(P)


def metricas_por_t(df, P):
    filas = []
    for t in CHECKPOINTS:
        m = (df["t"] == t).values
        y, p = df.loc[m, "y"].values, P[m]
        filas.append({"t": t,
                      "accuracy": accuracy_score(y, p.argmax(1)),
                      "log_loss": log_loss(y, p, labels=CLASES)})
    return pd.DataFrame(filas)


def baseline_prior(df, semilla, k=5):
    """Referencia trivial: predecir siempre las frecuencias de clase del train."""
    y, g = df["y"].values, df["fotmob_id"].values
    P = np.zeros((len(df), 3))
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla).split(y, y, g):
        frec = np.bincount(y[tr], minlength=3) / len(tr)
        P[te] = frec
    return suavizar(P)


def bootstrap_ic(df, P, n=500, seed=0):
    """IC 95% por bootstrap sobre partidos (no sobre filas)."""
    rng = np.random.default_rng(seed)
    ids = df["fotmob_id"].unique()
    idx_por_partido = df.groupby("fotmob_id").indices
    res = {t: {"accuracy": [], "log_loss": []} for t in CHECKPOINTS}
    for _ in range(n):
        muestra = rng.choice(ids, size=len(ids), replace=True)
        filas = np.concatenate([idx_por_partido[i] for i in muestra])
        sub, Ps = df.iloc[filas].reset_index(drop=True), P[filas]
        for t in CHECKPOINTS:
            m = (sub["t"] == t).values
            res[t]["accuracy"].append(accuracy_score(sub.loc[m, "y"], Ps[m].argmax(1)))
            res[t]["log_loss"].append(log_loss(sub.loc[m, "y"], Ps[m], labels=CLASES))
    out = []
    for t in CHECKPOINTS:
        out.append({"t": t,
                    "acc_lo": np.percentile(res[t]["accuracy"], 2.5),
                    "acc_hi": np.percentile(res[t]["accuracy"], 97.5),
                    "ll_lo": np.percentile(res[t]["log_loss"], 2.5),
                    "ll_hi": np.percentile(res[t]["log_loss"], 97.5)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data/processed/instances.csv")
    ap.add_argument("--salida", default="data/processed")
    ap.add_argument("--repeticiones", type=int, default=3)
    a = ap.parse_args()

    df = pd.read_csv(a.datos).sort_values(["fotmob_id", "t"]).reset_index(drop=True)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    assert df.isna().sum().sum() == 0, "hay valores faltantes"
    assert df.groupby("fotmob_id")["y"].nunique().max() == 1, "y debe ser constante por partido"
    print(f"{len(df)} instancias, {df.fotmob_id.nunique()} partidos")
    print("Distribución de y (por partido):",
          df.drop_duplicates("fotmob_id")["y"].value_counts().sort_index().to_dict())

    resultados = []

    # Referencia: frecuencias de clase
    Pr = np.mean([baseline_prior(df, s) for s in range(a.repeticiones)], axis=0)
    r = metricas_por_t(df, Pr)
    r["modelo"], r["variables"] = "Prior (frecuencias)", "-"
    resultados.append(r)

    ic_filas = []
    for nombre_conj, cols in CONJUNTOS.items():
        for nombre_mod, (mod, grilla) in modelos().items():
            # promediamos las probabilidades fuera de muestra de las repeticiones
            Ps = [evaluar(df, cols, mod, grilla, semilla) for semilla in range(a.repeticiones)]
            P = np.mean(Ps, axis=0)
            r = metricas_por_t(df, P)
            r["modelo"], r["variables"] = nombre_mod, nombre_conj
            resultados.append(r)
            ic = bootstrap_ic(df, P)
            ic["modelo"], ic["variables"] = nombre_mod, nombre_conj
            ic_filas.append(ic)
            print(f"{nombre_mod:7s} {nombre_conj:10s}  "
                  f"acc media={r.accuracy.mean():.3f}  logloss media={r.log_loss.mean():.3f}")

    res = pd.concat(resultados, ignore_index=True)
    ic = pd.concat(ic_filas, ignore_index=True)
    res = res.merge(ic, on=["t", "modelo", "variables"], how="left")
    res.to_csv(salida / "resultados.csv", index=False)

    # Tablas resumen
    for met in ["accuracy", "log_loss"]:
        tabla = res.pivot_table(index=["modelo", "variables"], columns="t", values=met)
        tabla["media"] = tabla.mean(axis=1)
        print(f"\n== {met} por checkpoint ==")
        print(tabla.round(3).to_string())
        tabla.round(3).to_csv(salida / f"tabla_{met}.csv")

    # Mejora respecto al baseline (log-loss; negativo = mejora)
    piv = res.pivot_table(index=["modelo", "t"], columns="variables", values="log_loss")
    mejora = pd.DataFrame({
        "d_juego_vs_base": piv["+juego"] - piv["base"],
        "d_contexto_vs_base": piv["+contexto"] - piv["base"],
    })
    print("\n== Δ log-loss respecto al modelo base (negativo = mejora) ==")
    print(mejora.round(3).to_string())
    mejora.round(4).to_csv(salida / "mejora_vs_base.csv")

    # Gráficos
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, met, titulo in zip(ejes, ["log_loss", "accuracy"], ["Log-loss (menor es mejor)", "Accuracy"]):
        for (mod, var), g in res[res.variables != "-"].groupby(["modelo", "variables"]):
            if var == "+contexto" or var == "base":
                ax.plot(g["t"], g[met], marker="o", ls="-" if var == "+contexto" else "--",
                        label=f"{mod} {var}")
        p = res[res.modelo == "Prior (frecuencias)"]
        ax.plot(p["t"], p[met], color="gray", ls=":", label="Prior")
        ax.set_xlabel("Minuto del checkpoint"); ax.set_title(titulo); ax.set_xticks(CHECKPOINTS)
    ejes[1].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(salida / "metricas_por_checkpoint.png", dpi=150)
    print(f"\nResultados guardados en {salida}/")


if __name__ == "__main__":
    main()