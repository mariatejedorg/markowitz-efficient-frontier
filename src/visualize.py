"""Visualización estática: nube de riesgo/retorno de los activos individuales
y los benchmarks, la frontera eficiente, y las carteras de referencia
(mínima varianza, máximo Sharpe, igual-ponderada) con la Capital Market Line.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.portfolio import RISK_FREE_RATE

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def plot_efficient_frontier(
    asset_names: list[str], mu: np.ndarray, Sigma: np.ndarray,
    frontier: pd.DataFrame, min_var_weights: np.ndarray, max_sharpe_weights: np.ndarray,
    equal_weights: np.ndarray, benchmark_points: dict[str, tuple[float, float]],
) -> None:
    from markowitz import portfolio_return, portfolio_volatility

    asset_vols = np.sqrt(np.diag(Sigma))

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(asset_vols, mu, color="tab:gray", s=50, zorder=3, label="Activos individuales")
    for name, vol, ret in zip(asset_names, asset_vols, mu):
        ax.annotate(name, (vol, ret), textcoords="offset points", xytext=(6, 4), fontsize=8, color="tab:gray")

    for name, (vol, ret) in benchmark_points.items():
        ax.scatter(vol, ret, color="black", marker="D", s=60, zorder=3)
        ax.annotate(name, (vol, ret), textcoords="offset points", xytext=(6, 4), fontsize=8, fontweight="bold")

    ax.plot(frontier["volatility"], frontier["target_return"], color="tab:blue", linewidth=2, label="Frontera eficiente", zorder=2)

    min_var_point = (portfolio_volatility(min_var_weights, Sigma), portfolio_return(min_var_weights, mu))
    max_sharpe_point = (portfolio_volatility(max_sharpe_weights, Sigma), portfolio_return(max_sharpe_weights, mu))
    equal_point = (portfolio_volatility(equal_weights, Sigma), portfolio_return(equal_weights, mu))

    ax.scatter(*min_var_point, color="tab:green", s=120, zorder=4, marker="*", label="Mínima varianza")
    ax.scatter(*max_sharpe_point, color="tab:red", s=120, zorder=4, marker="*", label="Máximo Sharpe (tangente)")
    ax.scatter(*equal_point, color="tab:orange", s=80, zorder=4, marker="s", label="Igual-ponderada")

    # Capital Market Line: recta desde (0, r_f) que pasa por la cartera tangente.
    cml_x = np.linspace(0, max_sharpe_point[0] * 1.4, 50)
    cml_slope = (max_sharpe_point[1] - RISK_FREE_RATE) / max_sharpe_point[0]
    cml_y = RISK_FREE_RATE + cml_slope * cml_x
    ax.plot(cml_x, cml_y, color="tab:red", linewidth=1, linestyle="--", label="Capital Market Line", zorder=1)

    ax.set_title("Frontera eficiente de Markowitz")
    ax.set_xlabel("Volatilidad anualizada")
    ax.set_ylabel("Retorno esperado anualizado")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "efficient_frontier.png", dpi=150)
    plt.close(fig)


def plot_portfolio_weights(asset_names: list[str], min_var_weights: np.ndarray,
                            max_sharpe_weights: np.ndarray, equal_weights: np.ndarray) -> None:
    """Barras agrupadas: composición de cada cartera de referencia, para ver
    qué tan concentrada sale la cartera de máximo Sharpe frente a la
    igual-ponderada."""
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(asset_names))
    width = 0.25

    ax.bar(x - width, min_var_weights * 100, width, color="tab:green", label="Mínima varianza")
    ax.bar(x, max_sharpe_weights * 100, width, color="tab:red", label="Máximo Sharpe")
    ax.bar(x + width, equal_weights * 100, width, color="tab:orange", label="Igual-ponderada")

    ax.set_xticks(x)
    ax.set_xticklabels(asset_names, rotation=0)
    ax.set_ylabel("Peso en la cartera (%)")
    ax.set_title("Composición de cada cartera de referencia")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "portfolio_weights.png", dpi=150)
    plt.close(fig)
