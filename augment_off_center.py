"""
Offline oversampling and augmentation for TEyeD gaze labels.

Given a preprocessed dataset (e.g., TEyeD_preprocessed/.../train),
this script copies the original images/labels to a new folder and
adds augmented copies of off-center gaze samples so training does
not need on-the-fly augmentation.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import pandas as pd


def off_center_info(df: pd.DataFrame, radius_threshold: float) -> tuple[np.ndarray, int, int, float]:
    """Return off-center mask, count, total, and percentage."""
    coords = df[["x", "y"]].to_numpy()
    radius = np.linalg.norm(coords, axis=1)
    off_center_mask = radius > radius_threshold
    off_cnt = int(off_center_mask.sum())
    total = len(df)
    pct = (off_cnt / total) * 100 if total else 0.0
    return off_center_mask, off_cnt, total, pct


def photometric_augment(image: np.ndarray) -> np.ndarray:
    """Apply light, label-safe augmentations to an image."""
    img = image.astype(np.float32)

    if random.random() < 0.7:
        alpha = random.uniform(0.9, 1.1)  # contrast
        beta = random.uniform(-12, 12)  # brightness shift in pixel domain
        img = img * alpha + beta

    if random.random() < 0.6:
        gamma = random.uniform(0.85, 1.2)
        img = 255.0 * np.power(np.clip(img / 255.0, 0.0, 1.0), gamma)

    if random.random() < 0.4:
        sigma = random.uniform(2.0, 6.0)
        noise = np.random.normal(0.0, sigma, img.shape)
        img = img + noise

    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    if random.random() < 0.6:
        delta = random.uniform(-0.1, 0.1)
        # BGR temperature shift: increase red, decrease blue (or vice versa).
        gains = np.array([1.0 - delta, 1.0, 1.0 + delta], dtype=np.float32)
        img = img * gains

    if random.random() < 0.4:
        amount = random.uniform(0.2, 0.45)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0.0)

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def copy_referenced_images(df: pd.DataFrame, src_split: Path, dst_split: Path) -> int:
    """Copy images referenced in labels.csv, returning how many were copied."""
    copied = 0
    for idx, row in df.iterrows():
        rel_path = Path(row["filename"])
        src_path = src_split / rel_path
        dst_path = dst_split / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not src_path.exists():
            print(f"Warning: missing image {src_path}")
            continue
        shutil.copy2(src_path, dst_path)
        copied += 1
        if copied % 5000 == 0:
            print(f"  Copied {copied} images...")
    return copied


def make_augmented_rows(
    df: pd.DataFrame,
    off_center_idx: Sequence[int],
    src_split: Path,
    dst_split: Path,
    oversample_factor: int,
    jpeg_quality: int,
) -> List[dict]:
    augmented = []
    off_center_df = df.loc[list(off_center_idx)]
    if off_center_df.empty or oversample_factor <= 1:
        return augmented

    for repeat in range(oversample_factor - 1):
        for _, row in off_center_df.iterrows():
            rel_path = Path(row["filename"])
            src_path = src_split / rel_path
            if not src_path.exists():
                print(f"Warning: missing image for augmentation {src_path}")
                continue

            image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: failed to read {src_path}")
                continue

            aug_image = photometric_augment(image)
            new_name = f"{rel_path.stem}_aug{repeat + 1}.jpg"
            new_rel_path = rel_path.with_name(new_name)
            dst_path = dst_split / new_rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(
                str(dst_path),
                aug_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not success:
                print(f"Warning: failed to write augmented image {dst_path}")
                continue

            augmented.append({"filename": str(new_rel_path), "x": row["x"], "y": row["y"]})
    return augmented


def oversample_split(
    split_name: str,
    split_dir: Path,
    out_dir: Path,
    radius_threshold: float,
    oversample_factor: int,
    seed: int,
    jpeg_quality: int,
    max_rows: int | None,
) -> None:
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        print(f"Skipping {split_name}: missing {labels_path}")
        return

    df = pd.read_csv(labels_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    off_center_mask, off_cnt, total_before, pct_before = off_center_info(df, radius_threshold)
    off_center_idx = np.where(off_center_mask)[0].tolist()
    print(
        f"[{split_name}] Before oversampling: total={total_before}, "
        f"off_center={off_cnt} ({pct_before:.2f}%)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_referenced_images(df, split_dir, out_dir)
    print(f"[{split_name}] Copied {copied} original images.")

    augmented_rows = make_augmented_rows(
        df,
        off_center_idx,
        split_dir,
        out_dir,
        oversample_factor=oversample_factor,
        jpeg_quality=jpeg_quality,
    )

    df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    coords_after = df_augmented[["x", "y"]].to_numpy()
    radius_after = np.linalg.norm(coords_after, axis=1)
    off_after = int((radius_after > radius_threshold).sum())
    pct_after = (off_after / len(df_augmented)) * 100 if len(df_augmented) else 0

    df_augmented.to_csv(out_dir / "labels.csv", index=False)
    print(
        f"[{split_name}] After oversampling: total={len(df_augmented)}, "
        f"off_center={off_after} ({pct_after:.2f}%)"
    )
    print(f"[{split_name}] Added {len(augmented_rows)} augmented rows.\n")


def copy_split_only(split_name: str, split_dir: Path, out_dir: Path, max_rows: int | None) -> None:
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        print(f"Skipping {split_name}: missing {labels_path}")
        return
    df = pd.read_csv(labels_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_referenced_images(df, split_dir, out_dir)
    df.to_csv(out_dir / "labels.csv", index=False)
    print(f"[{split_name}] Copied split without augmentation. images={copied}, rows={len(df)}\n")


def count_split_only(
    split_name: str, split_dir: Path, radius_threshold: float, max_rows: int | None
) -> None:
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        print(f"Skipping {split_name}: missing {labels_path}")
        return

    df = pd.read_csv(labels_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    _, off_cnt, total, pct = off_center_info(df, radius_threshold)
    print(f"[{split_name}] total={total}, off_center={off_cnt} ({pct:.2f}%)\n")


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) if args.output_root else None

    available_splits = []
    if (input_root / "labels.csv").exists():
        available_splits.append(input_root)
    available_splits.extend(
        [p for p in input_root.iterdir() if p.is_dir() and (p / "labels.csv").exists()]
    )
    if not available_splits:
        raise SystemExit(f"No split folders with labels.csv found under {input_root}")

    requested = set(args.splits) if args.splits else set()
    available_names = {p.name for p in available_splits}
    missing = requested - available_names
    if missing:
        print(f"Warning: requested splits not found: {sorted(missing)}\n")

    if args.count_only:
        print(f"Counting off-center samples under {input_root}\n")
        for split_dir in available_splits:
            split_name = split_dir.name
            if requested and split_name not in requested:
                continue
            count_split_only(
                split_name,
                split_dir,
                radius_threshold=args.radius_threshold,
                max_rows=args.max_rows,
            )
        return

    if output_root is None:
        raise SystemExit("--output_root is required unless --count_only is set")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Processing splits under {input_root} -> {output_root}")
    print(f"Augmenting: {sorted(requested) if requested else '[]'} (others will be copied only)\n")

    for split_dir in available_splits:
        split_name = split_dir.name
        if requested and split_name not in requested:
            copy_split_only(split_name, split_dir, output_root / split_name, max_rows=args.max_rows)
            continue

        out_dir = output_root / split_name
        oversample_split(
            split_name,
            split_dir,
            out_dir,
            radius_threshold=args.radius_threshold,
            oversample_factor=args.oversample_factor,
            seed=args.seed,
            jpeg_quality=args.jpeg_quality,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Oversample off-center gaze samples and bake in augmentation."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to preprocessed dataset root containing split folders (train/val/test).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=False,
        default=None,
        help="Where to write the copied + augmented dataset (required unless --count_only).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Split names to oversample (others will be copied unchanged).",
    )
    parser.add_argument("--radius_threshold", type=float, default=0.2)
    parser.add_argument("--oversample_factor", type=int, default=3)
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality for augmented images (0-100).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional limit per split for quick tests/trials.",
    )
    parser.add_argument(
        "--count_only",
        action="store_true",
        help="Only report how many off-center labels are present per split; no files are written.",
    )
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
