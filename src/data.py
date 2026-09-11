"""Descarga de precios históricos para varios tickers y cálculo de
retornos log diarios -- un ticker por columna."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.portfolio import HISTORICAL_PERIOD

# Mismo mecanismo que en los Proyectos 1-8: si un antivirus que inspecciona
# tráfico HTTPS (p. ej. Norton) rompe la verificación por defecto de yfinance,
# se usa un bundle de certificados local si existe.
_CUSTOM_CA_BUNDLE = Path(__file__).resolve().parent.parent / ".certs" / "cacert.pem"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _build_session():
    if not _CUSTOM_CA_BUNDLE.exists():
        return None
    from curl_cffi import requests as curl_requests

    return curl_requests.Session(impersonate="chrome", verify=str(_CUSTOM_CA_BUNDLE))


def download_prices(tickers: list[str], period: str = HISTORICAL_PERIOD) -> pd.DataFrame:
    """Precio de cierre ajustado diario para varios tickers, un ticker por columna."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, session=_build_session())
    prices = raw["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices.dropna(how="all").ffill().dropna()


def get_returns(tickers: list[str], period: str = HISTORICAL_PERIOD) -> pd.DataFrame:
    """Retornos logarítmicos diarios de todos los tickers, alineados por fecha."""
    prices = download_prices(tickers, period)
    return np.log(prices / prices.shift(1)).dropna()


def save_snapshot(prices: pd.DataFrame, name: str) -> Path:
    """Vuelca un DataFrame de precios a data/ como registro de la ejecución
    (no como caché de lectura)."""
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"prices_{name}.csv"
    prices.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    from config.portfolio import TICKERS

    returns = get_returns(list(TICKERS.keys()))
    print(f"Retornos descargados: {returns.shape[0]} días x {returns.shape[1]} activos")
    print(returns.tail())
