"""Dashboard HTML interactivo con Plotly: un único archivo autocontenido en
outputs/. Mismos tokens de color y helpers de layout que
proyecto-8-garch-volatilidad/src/dashboard.py, copiados literalmente para
dar continuidad visual entre proyectos del portfolio.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.portfolio import RISK_FREE_RATE

from markowitz import portfolio_return, portfolio_volatility

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

BLUE = "#2a78d6"
GREEN = "#1baf7a"
RED = "#e34948"
ORANGE = "#e3a648"
GRAY = "#898781"

SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(title: str, **extra) -> dict:
    layout = dict(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=15, color=INK_PRIMARY)),
        font=dict(family=FONT_FAMILY, size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        legend=dict(font=dict(color=INK_SECONDARY, size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    layout.update(extra)
    return layout


def _axis(**extra) -> dict:
    axis = dict(
        gridcolor=GRIDLINE,
        gridwidth=1,
        linecolor=BASELINE,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_SECONDARY, size=12),
        zeroline=False,
    )
    axis.update(extra)
    return axis


def _frontier_figure(
    asset_names: list[str], mu: np.ndarray, Sigma: np.ndarray, frontier: pd.DataFrame,
    min_var_weights: np.ndarray, max_sharpe_weights: np.ndarray, equal_weights: np.ndarray,
    benchmark_points: dict[str, tuple[float, float]],
) -> go.Figure:
    asset_vols = np.sqrt(np.diag(Sigma))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=asset_vols, y=mu, mode="markers+text", text=asset_names,
                              textposition="top center", name="Activos individuales",
                              marker=dict(color=GRAY, size=9)))

    if benchmark_points:
        bench_x = [p[0] for p in benchmark_points.values()]
        bench_y = [p[1] for p in benchmark_points.values()]
        fig.add_trace(go.Scatter(x=bench_x, y=bench_y, mode="markers+text", text=list(benchmark_points.keys()),
                                  textposition="top center", name="Benchmarks (no invertibles)",
                                  marker=dict(color=INK_PRIMARY, symbol="diamond", size=10)))

    fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["target_return"], mode="lines",
                              name="Frontera eficiente", line=dict(color=BLUE, width=2.5)))

    min_var_point = (portfolio_volatility(min_var_weights, Sigma), portfolio_return(min_var_weights, mu))
    max_sharpe_point = (portfolio_volatility(max_sharpe_weights, Sigma), portfolio_return(max_sharpe_weights, mu))
    equal_point = (portfolio_volatility(equal_weights, Sigma), portfolio_return(equal_weights, mu))

    fig.add_trace(go.Scatter(x=[min_var_point[0]], y=[min_var_point[1]], mode="markers", name="Mínima varianza",
                              marker=dict(color=GREEN, size=14, symbol="star")))
    fig.add_trace(go.Scatter(x=[max_sharpe_point[0]], y=[max_sharpe_point[1]], mode="markers", name="Máximo Sharpe",
                              marker=dict(color=RED, size=14, symbol="star")))
    fig.add_trace(go.Scatter(x=[equal_point[0]], y=[equal_point[1]], mode="markers", name="Igual-ponderada",
                              marker=dict(color=ORANGE, size=11, symbol="square")))

    cml_x = np.linspace(0, max_sharpe_point[0] * 1.4, 50)
    cml_slope = (max_sharpe_point[1] - RISK_FREE_RATE) / max_sharpe_point[0]
    fig.add_trace(go.Scatter(x=cml_x, y=RISK_FREE_RATE + cml_slope * cml_x, mode="lines",
                              name="Capital Market Line", line=dict(color=RED, width=1.2, dash="dash")))

    fig.update_layout(
        **_base_layout(
            "Frontera eficiente de Markowitz",
            xaxis=_axis(title="Volatilidad anualizada"),
            yaxis=_axis(title="Retorno esperado anualizado", tickformat=".0%"),
            hovermode="closest",
            hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK_PRIMARY, family=FONT_FAMILY)),
            height=500,
            xaxis_tickformat=".0%",
        )
    )
    return fig


def _weights_figure(asset_names: list[str], min_var_weights: np.ndarray,
                     max_sharpe_weights: np.ndarray, equal_weights: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=asset_names, y=min_var_weights * 100, name="Mínima varianza", marker=dict(color=GREEN)))
    fig.add_trace(go.Bar(x=asset_names, y=max_sharpe_weights * 100, name="Máximo Sharpe", marker=dict(color=RED)))
    fig.add_trace(go.Bar(x=asset_names, y=equal_weights * 100, name="Igual-ponderada", marker=dict(color=ORANGE)))

    fig.update_layout(
        **_base_layout(
            "Composición de cada cartera de referencia",
            xaxis=_axis(),
            yaxis=_axis(title="Peso en la cartera (%)"),
            barmode="group",
            height=400,
        )
    )
    return fig


def _stat_tile(label: str, value: str, sublabel: str) -> str:
    return f"""<div class="tile">
  <div class="tile-label">{label}</div>
  <div class="tile-value">{value}</div>
  <div class="tile-sublabel">{sublabel}</div>
</div>"""


def _kpi_tiles_html(min_var_metrics: dict, max_sharpe_metrics: dict, equal_metrics: dict) -> str:
    tiles = [
        _stat_tile("Mín. varianza — retorno/riesgo", f'{min_var_metrics["return"]*100:.1f}% / {min_var_metrics["vol"]*100:.1f}%', f'Sharpe {min_var_metrics["sharpe"]:.2f}'),
        _stat_tile("Máx. Sharpe — retorno/riesgo", f'{max_sharpe_metrics["return"]*100:.1f}% / {max_sharpe_metrics["vol"]*100:.1f}%', f'Sharpe {max_sharpe_metrics["sharpe"]:.2f}'),
        _stat_tile("Igual-ponderada — retorno/riesgo", f'{equal_metrics["return"]*100:.1f}% / {equal_metrics["vol"]*100:.1f}%', f'Sharpe {equal_metrics["sharpe"]:.2f}'),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def build_dashboard(
    asset_names: list[str], mu: np.ndarray, Sigma: np.ndarray, frontier: pd.DataFrame,
    min_var_weights: np.ndarray, max_sharpe_weights: np.ndarray, equal_weights: np.ndarray,
    benchmark_points: dict[str, tuple[float, float]],
    min_var_metrics: dict, max_sharpe_metrics: dict, equal_metrics: dict,
) -> Path:
    frontier_html = pio.to_html(
        _frontier_figure(asset_names, mu, Sigma, frontier, min_var_weights, max_sharpe_weights, equal_weights, benchmark_points),
        full_html=False, include_plotlyjs="cdn", config={"displaylogo": False},
    )
    weights_html = pio.to_html(
        _weights_figure(asset_names, min_var_weights, max_sharpe_weights, equal_weights),
        full_html=False, include_plotlyjs=False, config={"displaylogo": False},
    )
    tiles_html = _kpi_tiles_html(min_var_metrics, max_sharpe_metrics, equal_metrics)

    page = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proyecto 9 — Markowitz Portfolio Optimization</title>
<style>
  :root {{
    --surface: {SURFACE};
    --page-plane: {PAGE_PLANE};
    --ink-primary: {INK_PRIMARY};
    --ink-secondary: {INK_SECONDARY};
    --ink-muted: {INK_MUTED};
    --gridline: {GRIDLINE};
    --border: {BORDER};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: {FONT_FAMILY};
    margin: 0;
    background: var(--page-plane);
    color: var(--ink-primary);
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 24px 64px; }}

  .hero {{
    background: linear-gradient(135deg, #123a2e 0%, #1baf7a 100%);
    color: #ffffff;
    padding: 48px 24px 40px;
    margin-bottom: 28px;
  }}
  .hero-inner {{ max-width: 1080px; margin: 0 auto; }}
  .hero h1 {{ font-size: 1.75rem; margin: 0 0 8px; font-weight: 700; }}
  .hero p {{ margin: 0; color: rgba(255,255,255,0.85); font-size: 0.95rem; }}
  .hero .meta {{ margin-top: 18px; font-size: 0.8rem; color: rgba(255,255,255,0.65); letter-spacing: 0.02em; }}

  .tiles {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin: 0 0 28px;
  }}
  .tile {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }}
  .tile-label {{ font-size: 0.72rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .tile-value {{
    font-size: 1.45rem; font-weight: 700; color: var(--ink-primary); margin: 6px 0 2px;
    overflow-wrap: break-word;
  }}
  .tile-sublabel {{ font-size: 0.8rem; color: var(--ink-secondary); }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 22px;
  }}
  .card h2 {{ font-size: 1.05rem; margin: 0 0 16px; color: var(--ink-primary); font-weight: 600; }}

  footer {{ text-align: center; font-size: 0.78rem; color: var(--ink-muted); padding-top: 8px; }}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <h1>Markowitz Portfolio Optimization</h1>
    <p>De medir el riesgo a decidir la cartera: la frontera eficiente, la cartera de mínima varianza y la de máximo Sharpe, calculadas sobre 8 acciones europeas reales.</p>
    <div class="meta">Datos en vivo vía yfinance · misma cesta de activos que el Proyecto 1</div>
  </div>
</div>

<div class="wrap">

{tiles_html}

<div class="card">
  <h2>Frontera eficiente</h2>
  {frontier_html}
</div>

<div class="card">
  <h2>Composición de cada cartera</h2>
  {weights_html}
</div>

<footer>Proyecto 9 · Roadmap Quant · Python (numpy, scipy, pandas, yfinance, Plotly)</footer>

</div>
</body>
</html>"""

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / "dashboard.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path
