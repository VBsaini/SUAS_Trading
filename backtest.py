"""Event-based backtest for downloaded NIFTY OHLC CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ohlc_data(csv_path: str | Path) -> pd.DataFrame:
    """Load one daily OHLC series, averaging duplicate dates when needed."""
    data = pd.read_csv(csv_path)
    data.columns = [str(column).strip().lower() for column in data.columns]
    required = {"date", "open", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("open", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "close"])
    if data.empty:
        raise ValueError("CSV contains no valid OHLC rows")

    return (
        data.groupby("date", as_index=False)[["open", "close"]]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )


def run_backtest(
    df: pd.DataFrame,
    threshold_percent: float = 3.0,
    holding_period: int = 5,
    cost_percent: float = 0.10,
    max_fall_percent: float | None = None,
    min_fall_percent: float | None = None,
) -> dict[str, object]:
    """Buy after a sharp close-to-close fall and exit after N trading days."""
    required = {"date", "open", "close"}
    data = df.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Data is missing required columns: {sorted(missing)}")
    if threshold_percent <= 0 or holding_period < 1 or cost_percent < 0:
        raise ValueError("threshold and holding_period must be positive; cost cannot be negative")
    if max_fall_percent is not None and max_fall_percent <= threshold_percent:
        raise ValueError("max_fall_percent must be greater than threshold_percent")
    if min_fall_percent is not None and min_fall_percent < threshold_percent:
        raise ValueError("min_fall_percent must be at least threshold_percent")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["open"] = pd.to_numeric(data["open"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "open", "close"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    data["daily_return"] = data["close"].pct_change()

    threshold = threshold_percent / 100
    cost = cost_percent / 100
    max_fall = max_fall_percent / 100 if max_fall_percent is not None else None
    min_fall = min_fall_percent / 100 if min_fall_percent is not None else None
    trades: list[dict[str, object]] = []

    for signal_index, row in data.iterrows():
        if pd.isna(row["daily_return"]) or row["daily_return"] > -threshold:
            continue
        # Apply optional fall range filters
        if max_fall is not None and row["daily_return"] < -max_fall:
            continue
        if min_fall is not None and row["daily_return"] > -min_fall:
            continue
        entry_index = signal_index + 1
        exit_index = entry_index + holding_period - 1
        if exit_index >= len(data):
            continue

        entry_price = float(data.loc[entry_index, "open"])
        exit_price = float(data.loc[exit_index, "close"])
        if entry_price <= 0 or exit_price <= 0:
            continue

        gross_return = (exit_price - entry_price) / entry_price
        trades.append(
            {
                "signal_date": row["date"].strftime("%Y-%m-%d"),
                "previous_close": float(data.loc[signal_index - 1, "close"]),
                "signal_close": float(row["close"]),
                "daily_return_pct": float(row["daily_return"] * 100),
                "entry_date": data.loc[entry_index, "date"].strftime("%Y-%m-%d"),
                "exit_date": data.loc[exit_index, "date"].strftime("%Y-%m-%d"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return_pct": gross_return * 100,
                "net_return_pct": (gross_return - cost) * 100,
            }
        )

    if not trades:
        metrics = {
            "trade_count": 0,
            "average_return_pct": None,
            "median_return_pct": None,
            "win_rate_pct": None,
            "best_trade_pct": None,
            "worst_trade_pct": None,
        }
    else:
        returns = pd.Series([trade["net_return_pct"] for trade in trades], dtype=float)
        metrics = {
            "trade_count": len(trades),
            "average_return_pct": float(returns.mean()),
            "median_return_pct": float(returns.median()),
            "win_rate_pct": float((returns > 0).mean() * 100),
            "best_trade_pct": float(returns.max()),
            "worst_trade_pct": float(returns.min()),
        }

    return {"metrics": metrics, "trades": trades}


def run_csv_backtest(csv_path: str | Path, **kwargs: object) -> dict[str, object]:
    """Load a NIFTY CSV and run the event-based backtest."""
    return run_backtest(load_ohlc_data(csv_path), **kwargs)