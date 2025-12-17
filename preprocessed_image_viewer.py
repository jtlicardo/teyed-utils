import argparse
from pathlib import Path
import re

import matplotlib
import pandas as pd


def load_split(split_dir: Path) -> pd.DataFrame:
    """Load labels.csv and attach absolute image paths."""
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Point --split-dir at a folder containing labels.csv"
        )

    df = pd.read_csv(labels_path)
    if not {"filename", "x", "y"}.issubset(df.columns):
        raise ValueError("labels.csv must have columns: filename,x,y")

    df["path"] = df["filename"].apply(lambda name: split_dir / name)
    missing = df[~df["path"].apply(Path.exists)]
    if not missing.empty:
        print(
            f"Warning: {len(missing)} rows reference missing images. They will be skipped."
        )
        df = df[df["path"].apply(Path.exists)]

    if df.empty:
        raise RuntimeError("No images found after filtering missing files.")
    return df.reset_index(drop=True)


def label_to_pixel(
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    mapping: str,
    invert_y: bool,
    flip_x: bool,
) -> tuple[float, float]:
    """
    Convert a label (x,y) into pixel coordinates for drawing on the image.

    mapping:
      - "offset": interpret x,y in [-0.5, 0.5] and map into [0, W]x[0, H]
      - "vector": interpret x,y in [-1, 1] and draw an endpoint from the image center
    """
    if flip_x:
        x = -x
    if invert_y:
        y = -y

    if mapping == "offset":
        px = (x + 0.5) * float(width)
        py = (y + 0.5) * float(height)
        return px, py
    if mapping == "vector":
        cx, cy = width / 2.0, height / 2.0
        px = cx + x * (width / 2.0)
        py = cy + y * (height / 2.0)
        return px, py
    raise ValueError(f"Unknown mapping: {mapping}")


def infer_stride_from_path(split_dir: Path) -> int | None:
    for part in split_dir.parts[::-1]:
        m = re.search(r"stride(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def parse_image_index(image_path: Path) -> int | None:
    """
    Extract 1-based image index from a filename like 001879.jpg or 008790_aug2.jpg.
    """
    m = re.match(r"^(\d+)(?:_aug\d+)?$", image_path.stem)
    if not m:
        return None
    return int(m.group(1))


def infer_annotation_path(video_stem: str, annotation_root: Path) -> Path | None:
    candidates = [
        annotation_root / f"{video_stem}.mp4gaze_vec.txt",
        annotation_root / f"{video_stem.lower()}.mp4gaze_vec.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_gaze_vec_xy(annotation_path: Path) -> dict[int, tuple[float, float] | None]:
    """
    Load TEyeD gaze_vec into a frame->(x,y) dict. Frames are 1-based.
    Returns None for invalid (-1,-1,-1) frames.
    """
    out: dict[int, tuple[float, float] | None] = {}
    with annotation_path.open("r") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            frame = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if (x, y, z) == (-1.0, -1.0, -1.0):
                out[frame] = None
            else:
                out[frame] = (x, y)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive browser for TEyeD_preprocessed images. "
            "Displays ground-truth label with origin lines at the image center."
        )
    )
    parser.add_argument(
        "--split-dir",
        default="TEyeD_preprocessed/96x96_stride5_q4_train01/train",
        type=Path,
        help="Folder containing labels.csv and image subfolders (e.g., TEyeD_preprocessed/.../train).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the order of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed when shuffling.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="1-based index to start from (defaults to first sample).",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=None,
        help=(
            "Optional root folder containing TEyeD gaze_vec files. If omitted, tries "
            "`TEyeD/Dikablis/ANNOTATIONS` then `annotations`."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride used during preprocessing; inferred from split path when omitted.",
    )
    parser.add_argument(
        "--label-source",
        choices=["csv", "annotation", "both"],
        default="csv",
        help="Which label source to plot (default: csv).",
    )
    parser.add_argument(
        "--mapping",
        choices=["auto", "offset", "vector"],
        default="auto",
        help='How to map (x,y) into pixels: "offset" assumes [-0.5,0.5], "vector" assumes [-1,1].',
    )
    parser.add_argument(
        "--draw",
        choices=["point", "arrow", "both"],
        default="both",
        help="Draw a point marker, an arrow from the center, or both.",
    )
    parser.add_argument(
        "--flip-x",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flip the X sign when drawing (default: off).",
    )
    parser.add_argument(
        "--invert-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Invert Y sign when drawing (useful when comparing to Y-up plots; default: off)."
        ),
    )
    return parser.parse_args()


def ensure_interactive_backend() -> None:
    """Try to guarantee an interactive backend; error with guidance if not possible."""
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        return

    for candidate in ("MacOSX", "TkAgg", "Qt5Agg", "QtAgg"):
        try:
            matplotlib.use(candidate, force=True)
            print(f"Switched matplotlib backend to {candidate} for interactivity.")
            return
        except Exception:
            continue

    raise RuntimeError(
        "Matplotlib is using a non-interactive backend (Agg). "
        "Set an interactive backend, e.g. `MPLBACKEND=TkAgg uv run preprocessed_image_viewer.py ...`"
    )


def main() -> None:
    ensure_interactive_backend()
    import matplotlib.pyplot as plt

    args = parse_args()
    df = load_split(args.split_dir)
    stride = args.stride if args.stride is not None else infer_stride_from_path(args.split_dir)
    annotation_roots = (
        [args.annotation_root]
        if args.annotation_root is not None
        else [Path("TEyeD/Dikablis/ANNOTATIONS"), Path("annotations")]
    )
    gaze_cache: dict[Path, dict[int, tuple[float, float] | None]] = {}

    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    current = max(0, min(len(df) - 1, args.start - 1))
    records = df.to_dict("records")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    def resolve_annotation(video_stem: str) -> Path | None:
        for root in annotation_roots:
            if root is None:
                continue
            p = infer_annotation_path(video_stem, root)
            if p is not None:
                return p
        return None

    def get_annotation_xy(video_stem: str, img_index: int) -> tuple[float, float] | None:
        if stride is None:
            return None
        ann_path = resolve_annotation(video_stem)
        if ann_path is None:
            return None
        if ann_path not in gaze_cache:
            gaze_cache[ann_path] = load_gaze_vec_xy(ann_path)
        frame = 1 + (img_index - 1) * stride
        return gaze_cache[ann_path].get(frame)

    def choose_mapping(x: float, y: float) -> str:
        if args.mapping != "auto":
            return args.mapping
        return "vector" if max(abs(x), abs(y)) > 0.55 else "offset"

    def draw(idx: int) -> None:
        ax.clear()
        row = records[idx]
        img = plt.imread(row["path"])

        # Drop alpha if present to keep overlays readable.
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        height, width = img.shape[:2]
        center_x, center_y = width / 2.0, height / 2.0

        filename = str(row["filename"])
        video_stem = filename.split("/", 1)[0]
        img_index = parse_image_index(Path(filename))
        ann_xy = (
            get_annotation_xy(video_stem, img_index) if img_index is not None else None
        )

        csv_xy = (float(row["x"]), float(row["y"]))
        plotted: list[tuple[str, tuple[float, float] | None]] = []
        if args.label_source in ("csv", "both"):
            plotted.append(("CSV", csv_xy))
        if args.label_source in ("annotation", "both"):
            plotted.append(("ANN", ann_xy))

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, origin="upper")
        ax.axvline(center_x, color="cyan", linestyle="--", alpha=0.35)
        ax.axhline(center_y, color="cyan", linestyle="--", alpha=0.35)
        ax.scatter([center_x], [center_y], c="cyan", s=50, marker="o", label="Origin")

        for label_name, xy in plotted:
            if xy is None:
                continue
            x, y = xy
            mapping = choose_mapping(x, y)
            px, py = label_to_pixel(
                x,
                y,
                width,
                height,
                mapping=mapping,
                invert_y=args.invert_y,
                flip_x=args.flip_x,
            )
            if args.draw in ("arrow", "both"):
                ax.arrow(
                    center_x,
                    center_y,
                    px - center_x,
                    py - center_y,
                    color="lime" if label_name == "CSV" else "magenta",
                    linewidth=2.0,
                    head_width=max(2.0, width * 0.02),
                    length_includes_head=True,
                    alpha=0.9,
                )
            if args.draw in ("point", "both"):
                ax.scatter(
                    [px],
                    [py],
                    c="lime" if label_name == "CSV" else "magenta",
                    s=120,
                    marker="+",
                    linewidths=2.0,
                    label=label_name,
                )

        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)  # top-left origin
        ax.set_aspect("equal")
        ax.set_xlabel("X (pixels, origin at top-left)")
        ax.set_ylabel("Y (pixels, origin at top-left)")
        ax.set_title(f"{idx + 1}/{len(records)} — {row['filename']}")
        ax.legend(loc="upper right", fontsize="small")

        ann_text = "N/A"
        if stride is None:
            ann_text = "stride unknown"
        elif img_index is None:
            ann_text = "unparsed filename"
        elif ann_xy is None:
            ann_text = "missing/invalid"
        else:
            ann_text = f"({ann_xy[0]:.3f}, {ann_xy[1]:.3f})"

        mapping_used = choose_mapping(csv_xy[0], csv_xy[1])
        ax.text(
            5,
            20,
            f"CSV:   ({csv_xy[0]:.3f}, {csv_xy[1]:.3f})\n"
            f"ANN:   {ann_text}\n"
            f"Mode:  label={args.label_source}, map={args.mapping}({mapping_used}), "
            f"flip_x={'on' if args.flip_x else 'off'}, inv_y={'on' if args.invert_y else 'off'}",
            color="white",
            fontsize=9,
            fontweight="bold",
            fontfamily="monospace",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
        )

        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal current
        if event.key == "x":
            args.flip_x = not args.flip_x
            draw(current)
            return
        if event.key == "y":
            args.invert_y = not args.invert_y
            draw(current)
            return
        if event.key == "l":
            order = ["csv", "annotation", "both"]
            args.label_source = order[(order.index(args.label_source) + 1) % len(order)]
            draw(current)
            return
        if event.key == "m":
            order = ["auto", "offset", "vector"]
            args.mapping = order[(order.index(args.mapping) + 1) % len(order)]
            draw(current)
            return
        if event.key == "d":
            order = ["both", "point", "arrow"]
            args.draw = order[(order.index(args.draw) + 1) % len(order)]
            draw(current)
            return
        if event.key in ("q", "escape"):
            plt.close(fig)
            return
        if event.key in ("right", "n", " ", "enter"):
            current = (current + 1) % len(records)
        elif event.key in ("left", "p", "backspace"):
            current = (current - 1) % len(records)
        elif event.key == "home":
            current = 0
        elif event.key == "end":
            current = len(records) - 1
        draw(current)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(
        "Controls: right/left or n/p to step; space/enter next; home/end jump; "
        "l label-src; m mapping; d draw; x flip-x; y invert-y; q/esc quit."
    )
    draw(current)
    plt.show(block=True)


if __name__ == "__main__":
    main()
