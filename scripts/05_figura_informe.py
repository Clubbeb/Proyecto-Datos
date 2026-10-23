"""
Figura del informe: log-loss de la Regresión Logística por checkpoint, con
intervalos de confianza del 95 % (bootstrap por partido), para los tres
conjuntos de variables, y la referencia por frecuencias de clase.

Lee results/resultados.csv (generado por scripts/04_modelos.py).

Uso (desde la raíz del repo):
    python scripts/05_figura_informe.py
    python scripts/05_figura_informe.py --salida report/fig_logloss_ic.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ESTILO = {  # conjunto: (etiqueta, color, desplazamiento en el eje x)
    "base":      ("base (solo marcador)", "#8c8c8c", -1.2),
    "+juego":    ("+juego (xG y tiros)",  "#c4c4c4",  0.0),
    "+contexto": ("+contexto",            "#1f6fd1",  1.2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="results/resultados.csv")
    ap.add_argument("--salida", default="report/fig_logloss_ic.png")
    a = ap.parse_args()

    res = pd.read_csv(a.resultados)
    prior = res[res.modelo.str.startswith("Prior")].log_loss.iloc[0]
    rl = res[res.modelo == "RegLog"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for var, (etiqueta, color, dx) in ESTILO.items():
        g = rl[rl.variables == var].sort_values("t")
        ax.errorbar(g.t + dx, g.log_loss,
                    yerr=[g.log_loss - g.ll_lo, g.ll_hi - g.log_loss],
                    marker="o", ms=5, lw=2 if var == "+contexto" else 1.4,
                    capsize=3, color=color, label=etiqueta, zorder=3 if var == "+contexto" else 2)
    ax.axhline(prior, ls="--", color="#555555", lw=1)
    ax.text(76.5, prior + 0.012, f"Prior (frecuencias): {prior:.3f}".replace(".", ","),
            ha="right", va="bottom", fontsize=8, color="#555555")

    ax.set_xticks([15, 30, 45, 60, 75])
    ax.set_xticklabels(["15'", "30'", "45'", "60'", "75'"])
    ax.set_xlabel("Minuto del checkpoint")
    ax.set_ylabel("Log-loss (menor es mejor)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.grid(axis="y", alpha=0.3)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=200, bbox_inches="tight")
    print(f"Figura guardada en {salida}")


if __name__ == "__main__":
    main()