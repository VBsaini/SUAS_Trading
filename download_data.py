"""Download a Kaggle dataset and create the CSV consumed by the API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import kagglehub
import pandas as pd
from dotenv import load_dotenv

REQUIRED = {"date", "open", "close"}
ALIASES = {
    "timestamp": "date",
    "datetime": "date",
    "time": "date",
    "adj close": "close",
    "adj_close": "close",
}


def _normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in data.columns:
        name = str(column).strip().lower().replace("-", " ")
        renamed[column] = ALIASES.get(name, name.replace(" ", "_"))
    return data.rename(columns=renamed)


def _find_csv(download_path: Path) -> Path:
    candidates = sorted(download_path.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Kaggle download contains no CSV files: {download_path}")
    for path in candidates:
        columns = set(_normalise_columns(pd.read_csv(path, nrows=0)).columns)
        if REQUIRED <= columns:
            return path
    raise ValueError("Kaggle download contains no CSV with date, open, and close columns")


def convert_to_ohlc(source: Path, destination: Path) -> int:
    data = _normalise_columns(pd.read_csv(source))
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("open", "high", "low", "close"):
        if column not in data:
            data[column] = data["close"] if column != "open" else data["close"]
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "close"])
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    data = (
        data[["date", "open", "high", "low", "close"]]
        .groupby("date", as_index=False)
        .mean()
        .sort_values("date")
    )
    if data.empty:
        raise ValueError("CSV contains no valid OHLC rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    data.to_csv(destination, index=False)
    return len(data)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("nifty50", "nifty250"), help="Use the matching dataset and output settings from .env")
    parser.add_argument("--dataset", help="Kaggle dataset handle, such as owner/dataset")
    parser.add_argument("--output", help="Destination CSV path")
    args = parser.parse_args()
    if args.market:
        prefix = "NIFTY50" if args.market == "nifty50" else "NIFTY250"
        args.dataset = args.dataset or os.getenv(f"{prefix}_DATASET")
        args.output = args.output or os.getenv(f"{prefix}_DATA_CSV_PATH")
    args.output = args.output or "data/nifty_ohlc.csv"
    if not args.dataset:
        parser.error("pass --dataset or set a market dataset in .env")
    downloaded = Path(kagglehub.dataset_download(args.dataset))
    source = _find_csv(downloaded)
    rows = convert_to_ohlc(source, Path(args.output))
    print(f"Downloaded: {args.dataset}")
    print(f"Source CSV: {source}")
    print(f"Output CSV: {Path(args.output).resolve()} ({rows} rows)")


if __name__ == "__main__":
    main()