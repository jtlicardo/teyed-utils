"""
Gaze density plotter for TEyeD-style labels.

Point this at a folder containing labels.csv and it will
render a hexbin density plot of x/y gaze coordinates.

Example: uv run plot_gaze_density.py --input_root path/to/folder
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_labels(split_dir: Path, max_rows: int | None) -> pd.DataFrame:
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found under {split_dir}")

    df = pd.read_csv(labels_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df


def plot_density(df: pd.DataFrame, gridsize: int, mincnt: int, title: str | None) -> None:
    plt.figure(figsize=(6, 6))
    plt.hexbin(df["x"], df["y"], gridsize=gridsize, cmap="viridis", mincnt=mincnt)
    plt.colorbar(label="count")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title or "2D gaze density")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a hexbin density of gaze labels from labels.csv."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Folder containing labels.csv (e.g., TEyeD_preprocessed/train).",
    )
    parser.add_argument("--gridsize", type=int, default=60, help="Hexbin grid resolution.")
    parser.add_argument("--mincnt", type=int, default=1, help="Minimum count to shade a bin.")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional limit for a quick preview without loading all rows.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title; defaults to '2D gaze density'.",
    )
    args = parser.parse_args()

    split_dir = Path(args.input_root)
    df = load_labels(split_dir, max_rows=args.max_rows)
    if df.empty:
        raise SystemExit(f"No rows found in {split_dir / 'labels.csv'}")

    plot_density(df, gridsize=args.gridsize, mincnt=args.mincnt, title=args.title)


if __name__ == "__main__":
    main()
