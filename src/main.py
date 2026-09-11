"""Punto de entrada: descarga retornos reales del universo invertible y de
los benchmarks, calcula la frontera eficiente y las carteras de referencia
(mínima varianza, máximo Sharpe, igual-ponderada), y genera las
visualizaciones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.portfolio import BENCHMARK_TICKERS, RISK_FREE_RATE, TICKERS

from dashboard import build_dashboard
from data import get_returns, save_snapshot
from markowitz import (
    annualize_mean_cov,
    efficient_frontier,
    equal_weight_portfolio,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)
from visualize import plot_efficient_frontier, plot_portfolio_weights


def _metrics(weights, mu, Sigma):
    return {
        "return": portfolio_return(weights, mu),
        "vol": portfolio_volatility(weights, Sigma),
        "sharpe": sharpe_ratio(weights, mu, Sigma),
    }


def main() -> None:
    asset_names = list(TICKERS.keys())
    returns = get_returns(asset_names)
    save_snapshot(returns, "universe")

    benchmark_returns = get_returns(list(BENCHMARK_TICKERS.keys()))
    save_snapshot(benchmark_returns, "benchmarks")

    print("=== Datos descargados ===")
    print(f"Universo invertible: {len(asset_names)} activos | Observaciones: {len(returns)}")

    mu, Sigma = annualize_mean_cov(returns)
    bench_mu, bench_Sigma = annualize_mean_cov(benchmark_returns)
    benchmark_points = {
        name: (float((bench_Sigma[i, i]) ** 0.5), float(bench_mu[i]))
        for i, name in enumerate(BENCHMARK_TICKERS.values())
    }

    min_var_weights = min_variance_portfolio(mu, Sigma)
    max_sharpe_weights = max_sharpe_portfolio(mu, Sigma, RISK_FREE_RATE)
    equal_weights = equal_weight_portfolio(len(asset_names))
    frontier = efficient_frontier(mu, Sigma)

    min_var_metrics = _metrics(min_var_weights, mu, Sigma)
    max_sharpe_metrics = _metrics(max_sharpe_weights, mu, Sigma)
    equal_metrics = _metrics(equal_weights, mu, Sigma)

    print("\n=== Carteras de referencia (retorno / volatilidad / Sharpe, anualizados) ===")
    print(f'Mínima varianza:  {min_var_metrics["return"]:+.2%} / {min_var_metrics["vol"]:.2%} / {min_var_metrics["sharpe"]:.2f}')
    print(f'Máximo Sharpe:    {max_sharpe_metrics["return"]:+.2%} / {max_sharpe_metrics["vol"]:.2%} / {max_sharpe_metrics["sharpe"]:.2f}')
    print(f'Igual-ponderada:  {equal_metrics["return"]:+.2%} / {equal_metrics["vol"]:.2%} / {equal_metrics["sharpe"]:.2f}')

    print("\n=== Composición de la cartera de máximo Sharpe ===")
    for name, weight in zip(TICKERS.values(), max_sharpe_weights):
        if weight > 0.01:
            print(f"{name}: {weight:.1%}")

    print("\n=== Benchmarks (no invertibles, solo referencia) ===")
    for name, (vol, ret) in benchmark_points.items():
        print(f"{name}: retorno {ret:+.2%} / volatilidad {vol:.2%}")

    plot_efficient_frontier(list(TICKERS.values()), mu, Sigma, frontier, min_var_weights, max_sharpe_weights, equal_weights, benchmark_points)
    plot_portfolio_weights(list(TICKERS.values()), min_var_weights, max_sharpe_weights, equal_weights)
    dashboard_path = build_dashboard(
        list(TICKERS.values()), mu, Sigma, frontier, min_var_weights, max_sharpe_weights, equal_weights,
        benchmark_points, min_var_metrics, max_sharpe_metrics, equal_metrics,
    )

    print(f"\nGráficos guardados en outputs/ (dashboard interactivo: {dashboard_path})")


if __name__ == "__main__":
    main()
