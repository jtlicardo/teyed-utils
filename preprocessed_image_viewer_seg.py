import argparse
from pathlib import Path

import matplotlib
import pandas as pd


SEG_PART_CHOICES = ("pupil", "iris", "lid")
SEG_PART_CMAPS = {
    "pupil": "Reds",
    "iris": "Greens",
    "lid": "Blues",
}


def resolve_seg_parts(seg_parts: str) -> tuple[str, ...]:
    raw = seg_parts.strip().lower()
    if raw == "all":
        return SEG_PART_CHOICES

    parts = []
    for token in seg_parts.split(","):
        part = token.strip().lower()
        if not part:
            continue
        if part not in SEG_PART_CHOICES:
            raise ValueError(
                f"Invalid segmentation part '{part}'. Choose from {SEG_PART_CHOICES}."
            )
        if part not in parts:
            parts.append(part)

    if not parts:
        raise ValueError(
            "No valid segmentation parts were provided. "
            "Use --seg_parts pupil,iris,lid or --seg_parts all."
        )
    return tuple(parts)


def load_masks(split_dir: Path, parts: tuple[str, ...], dimension: str) -> pd.DataFrame:
    """Load labels.csv and attach absolute image + mask paths."""
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Point --split-dir at a folder containing labels.csv"
        )

    df = pd.read_csv(labels_path)
    if "filename" not in df.columns:
        raise ValueError("labels.csv must have a filename column")

    df["image_path"] = df["filename"].apply(lambda name: split_dir / name)
    for part in parts:
        seg_root = split_dir / f"seg_{part}_{dimension}"
        if not seg_root.exists():
            raise FileNotFoundError(
                f"{seg_root} not found. Run preprocess_teyed.py with --segmentation "
                "or point --split-dir at a split containing those masks."
            )

        def mask_path(name: str, root: Path = seg_root) -> Path:
            rel = Path(name).with_suffix(".png")
            return root / rel

        df[f"mask_path_{part}"] = df["filename"].apply(mask_path)

    missing_images = df[~df["image_path"].apply(Path.exists)]
    if not missing_images.empty:
        print(
            f"Warning: {len(missing_images)} rows reference missing images. "
            "They will be skipped."
        )
        df = df[df["image_path"].apply(Path.exists)]

    for part in parts:
        col = f"mask_path_{part}"
        missing_masks = df[~df[col].apply(Path.exists)]
        if not missing_masks.empty:
            print(
                f"Warning: {len(missing_masks)} rows reference missing {part} masks. "
                "They will be skipped."
            )
            df = df[df[col].apply(Path.exists)]

    if df.empty:
        raise RuntimeError("No masks found after filtering missing files.")
    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive browser for TEyeD_preprocessed segmentation masks."
    )
    parser.add_argument(
        "--split-dir",
        default="TEyeD_preprocessed/96x96_stride5_q4_train01_seg_pupil-iris-lid_2D/train",
        type=Path,
        help="Folder containing labels.csv and seg_* folders (e.g., TEyeD_preprocessed/.../train).",
    )
    parser.add_argument(
        "--seg_parts",
        type=str,
        default="pupil,iris,lid",
        help=(
            "Comma-separated parts to display together "
            "(e.g., pupil,iris,lid) or 'all'."
        ),
    )
    parser.add_argument(
        "--seg_dimension",
        choices=["2D", "3D", "3Dplane"],
        default="2D",
        help="Which segmentation dimension to display.",
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
        "--binarize",
        action="store_true",
        help="Binarize masks for display (threshold at 0.5).",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the mask on top of the source image (default: on).",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Mask overlay alpha (0 to 1).",
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
        "Set an interactive backend, e.g. `MPLBACKEND=TkAgg uv run preprocessed_image_viewer_seg.py ...`"
    )


def main() -> None:
    ensure_interactive_backend()
    import matplotlib.pyplot as plt

    args = parse_args()
    parts = resolve_seg_parts(args.seg_parts)
    df = load_masks(args.split_dir, parts, args.seg_dimension)

    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    current = max(0, min(len(df) - 1, args.start - 1))
    records = df.to_dict("records")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    def normalize_mask(mask):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        max_val = float(mask.max()) if mask.size else 1.0
        if max_val > 1.0:
            mask = mask / 255.0
        if args.binarize:
            mask = (mask > 0.5).astype("float32")
        return mask

    def draw(idx: int) -> None:
        ax.clear()
        row = records[idx]
        img = plt.imread(row["image_path"])
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        masks = {}
        for part in parts:
            mask = plt.imread(row[f"mask_path_{part}"])
            masks[part] = normalize_mask(mask)

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, origin="upper")
        if args.overlay:
            alpha = max(0.0, min(1.0, args.mask_alpha))
            # Draw pupil last so it stays visually on top of other masks.
            overlay_parts = [part for part in parts if part != "pupil"]
            if "pupil" in parts:
                overlay_parts.append("pupil")

            for part in overlay_parts:
                mask = masks[part]
                ax.imshow(
                    mask,
                    cmap=SEG_PART_CMAPS.get(part, "autumn"),
                    origin="upper",
                    vmin=0.0,
                    vmax=1.0,
                    alpha=mask * alpha,
                )

        ax.set_title(f"{idx + 1}/{len(records)} — {row['filename']}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_aspect("equal")

        part_lines = []
        for part in parts:
            mask = masks[part]
            white_ratio = float(mask.mean()) if mask.size else 0.0
            part_lines.append(f"{part:>5} ({SEG_PART_CMAPS.get(part, 'n/a')}): {white_ratio:.4f}")

        ax.text(
            5,
            20,
            f"parts: {','.join(parts)}\n"
            f"binarize: {'on' if args.binarize else 'off'}\n"
            f"overlay: {'on' if args.overlay else 'off'} (alpha={args.mask_alpha:.2f})\n"
            + "white_ratio:\n"
            + "\n".join(part_lines),
            color="white",
            fontsize=9,
            fontweight="bold",
            fontfamily="monospace",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
        )

        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal current
        if event.key in ("q", "escape"):
            plt.close(fig)
            return
        if event.key == "b":
            args.binarize = not args.binarize
            draw(current)
            return
        if event.key == "o":
            args.overlay = not args.overlay
            draw(current)
            return
        if event.key == "]":
            args.mask_alpha = min(1.0, args.mask_alpha + 0.05)
            draw(current)
            return
        if event.key == "[":
            args.mask_alpha = max(0.0, args.mask_alpha - 0.05)
            draw(current)
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
        "b binarize; o overlay; [/] alpha; q/esc quit."
    )
    draw(current)
    plt.show(block=True)


if __name__ == "__main__":
    main()
