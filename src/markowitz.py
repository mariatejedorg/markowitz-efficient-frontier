"""Optimización de carteras de Markowitz (1952): dado el retorno esperado y
la matriz de covarianzas de un conjunto de activos, calcula la cartera de
mínima varianza, la frontera eficiente completa, y la cartera de máximo
Sharpe (la cartera tangente con el activo libre de riesgo).

Las fórmulas de retorno y riesgo de una cartera están escritas a mano; la
optimización numérica en sí (minimizar una función sujeta a restricciones)
se delega en scipy.optimize.minimize -- igual que el resto del portfolio
usa scipy para la pieza puramente numérica y escribe a mano la pieza que
es el modelo financiero.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.portfolio import N_FRONTIER_POINTS, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def annualize_mean_cov(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """mu: retorno esperado anualizado de cada activo (media de retornos log
    diarios x 252). Sigma: matriz de covarianzas anualizada (x 252) -- la
    varianza y la covarianza de sumas de variables independientes en el
    tiempo escalan linealmente con el número de días, de ahí el mismo
    factor 252 que en la anualización de sigma en Proyectos 1/3/8."""
    mu = returns.mean().to_numpy() * TRADING_DAYS_PER_YEAR
    Sigma = returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    return mu, Sigma


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """R(w) = w' * mu."""
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, Sigma: np.ndarray) -> float:
    """sigma(w) = sqrt(w' * Sigma * w)."""
    return float(np.sqrt(weights @ Sigma @ weights))


def _equal_constraint_sum_to_one(weights: np.ndarray) -> float:
    return np.sum(weights) - 1.0


def min_variance_portfolio(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Pesos que minimizan la varianza de la cartera, sujeto a sum(w)=1 y
    0<=w<=1 (long-only: sin ventas en corto, la restricción práctica más
    habitual para un primer ejercicio de Markowitz)."""
    n = len(mu)
    initial_weights = np.full(n, 1 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": _equal_constraint_sum_to_one}]

    result = minimize(
        lambda w: portfolio_volatility(w, Sigma) ** 2, initial_weights,
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x


def efficient_frontier(mu: np.ndarray, Sigma: np.ndarray, n_points: int = N_FRONTIER_POINTS) -> pd.DataFrame:
    """Traza la frontera eficiente: para cada retorno objetivo entre el de
    la cartera de mínima varianza y el del activo con mayor retorno
    esperado, encuentra la cartera de menor riesgo que lo alcanza.

    Por debajo del retorno de mínima varianza la frontera "eficiente" no
    tiene sentido (para ese riesgo hay una cartera con más retorno) --
    por eso el rango de la rejilla empieza justo en ese punto.
    """
    n = len(mu)
    min_var_weights = min_variance_portfolio(mu, Sigma)
    min_return = portfolio_return(min_var_weights, mu)
    max_return = mu.max()

    target_returns = np.linspace(min_return, max_return, n_points)
    bounds = [(0.0, 1.0)] * n

    rows = []
    initial_weights = np.full(n, 1 / n)
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": _equal_constraint_sum_to_one},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu) - t},
        ]
        result = minimize(
            lambda w: portfolio_volatility(w, Sigma) ** 2, initial_weights,
            method="SLSQP", bounds=bounds, constraints=constraints,
        )
        rows.append({"target_return": target, "volatility": portfolio_volatility(result.x, Sigma), **{
            f"w_{i}": result.x[i] for i in range(n)
        }})

    return pd.DataFrame(rows)


def max_sharpe_portfolio(mu: np.ndarray, Sigma: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> np.ndarray:
    """Pesos que maximizan el Sharpe ratio (R(w) - r_f) / sigma(w) --
    equivalente a minimizar su negativo, sujeto a las mismas restricciones
    long-only que min_variance_portfolio."""
    n = len(mu)
    initial_weights = np.full(n, 1 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": _equal_constraint_sum_to_one}]

    def neg_sharpe(w):
        excess_return = portfolio_return(w, mu) - risk_free_rate
        return -excess_return / portfolio_volatility(w, Sigma)

    result = minimize(neg_sharpe, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """Cartera igual-ponderada (1/N cada activo): referencia ingenua de
    comparación, sin usar ninguna información de mu ni Sigma."""
    return np.full(n_assets, 1 / n_assets)


def sharpe_ratio(weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    return (portfolio_return(weights, mu) - risk_free_rate) / portfolio_volatility(weights, Sigma)


if __name__ == "__main__":
    # Verificación contra una fórmula cerrada de libro de texto: para 2
    # activos, la cartera de mínima varianza tiene solución analítica
    #
    #   w1* = (sigma2^2 - rho*sigma1*sigma2) / (sigma1^2 + sigma2^2 - 2*rho*sigma1*sigma2)
    #
    # Se compara ese w1* calculado a mano contra min_variance_portfolio()
    # sobre los mismos dos activos sintéticos.
    sigma1, sigma2, rho = 0.20, 0.30, 0.40
    mu_synthetic = np.array([0.08, 0.12])  # los retornos esperados no entran en la fórmula de mín. varianza

    cov12 = rho * sigma1 * sigma2
    Sigma_synthetic = np.array([[sigma1**2, cov12], [cov12, sigma2**2]])

    w1_closed_form = (sigma2**2 - cov12) / (sigma1**2 + sigma2**2 - 2 * cov12)

    weights_numeric = min_variance_portfolio(mu_synthetic, Sigma_synthetic)

    print(f"w1 (fórmula cerrada) = {w1_closed_form:.4f}")
    print(f"w1 (scipy.optimize)  = {weights_numeric[0]:.4f}")
    print(f"w2 (scipy.optimize)  = {weights_numeric[1]:.4f} (debería ser 1 - w1)")
