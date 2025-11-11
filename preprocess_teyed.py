import argparse
import random
import subprocess
from pathlib import Path

import pandas as pd


def extract_video_ffmpeg(
    video_path: Path, out_dir: Path, size: tuple, stride: int, q: int
) -> int:
    """Extracts frames from a video using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    vf = f"select='not(mod(n\\,{stride}))',setpts=N/FRAME_RATE/TB,scale={w}:{h}"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-q:v",
        str(q),
        "-start_number",
        "1",
        str(out_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.jpg")))


def build_split_labels(
    video_names, videos_dir, annotations_dir, out_root, frame_size, frame_stride, jpeg_q
) -> Path:
    """Processes a list of videos, extracts frames, and creates a labels.csv file."""
    rows = []
    print(f"Processing {len(video_names)} videos for split: {out_root.name}...")

    for i, name in enumerate(video_names):
        vpath = videos_dir / name
        apath = annotations_dir / f"{name}gaze_vec.txt"
        if not (vpath.exists() and apath.exists()):
            print(f"  [{i + 1}/{len(video_names)}] Skip missing: {name}")
            continue

        stem = vpath.stem
        out_dir = out_root / stem
        saved = extract_video_ffmpeg(
            vpath, out_dir, size=frame_size, stride=frame_stride, q=jpeg_q
        )

        df = pd.read_csv(
            apath,
            sep=";",
            engine="python",
            usecols=[0, 1, 2],
            names=["frame", "x", "y"],
            header=0,
        )
        if frame_stride > 1:
            df = df.iloc[::frame_stride]  # downsample labels

        n = min(saved, len(df))
        if n == 0:
            print(f"  [{i + 1}/{len(video_names)}] No frames/labels for {stem}")
            continue

        df = df.iloc[:n]

        file_names = [f"{stem}/{j:06d}.jpg" for j in range(1, n + 1)]
        rows.extend(zip(file_names, df["x"].to_numpy(), df["y"].to_numpy()))
        print(
            f"  [{i + 1}/{len(video_names)}] {stem}: frames_saved={saved}, labels_used={n}"
        )

    out_csv = out_root / "labels.csv"
    pd.DataFrame(rows, columns=["filename", "x", "y"]).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(rows)} rows\n")
    return out_csv


def main(args):
    """Main function to run the preprocessing pipeline."""
    random.seed(args.seed)

    # Define paths
    root = Path(args.data_root)
    assert root.exists(), f"Missing raw data root: {root}"

    videos_dir = root / "VIDEOS"
    annotations_dir = root / "ANNOTATIONS"

    output_parent = Path(args.output_root)
    params_name = f"{args.frame_width}x{args.frame_height}_stride{args.frame_stride}"
    output_base = output_parent / params_name
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Saving preprocessed data to: {output_base}")

    repo_dir = Path(args.splits_dir)
    split_files = {
        "train": repo_dir / "train.txt",
        "val": repo_dir / "val.txt",
        "test": repo_dir / "test.txt",
    }
    splits = {}
    for split, file_path in split_files.items():
        with open(file_path) as f:
            splits[split] = [line.strip() for line in f if line.strip()]

    # Sample splits if fractions are provided
    if args.train_sample_frac < 1.0:
        num_to_sample = max(1, int(len(splits["train"]) * args.train_sample_frac))
        splits["train"] = random.sample(splits["train"], num_to_sample)
        print(f"Sampled {len(splits['train'])} videos for training.")

    if args.val_sample_frac < 1.0:
        num_to_sample = max(1, int(len(splits["val"]) * args.val_sample_frac))
        splits["val"] = random.sample(splits["val"], num_to_sample)
        print(f"Sampled {len(splits['val'])} videos for validation.")

    # Process each split
    frame_params = {
        "frame_size": (args.frame_width, args.frame_height),
        "frame_stride": args.frame_stride,
        "jpeg_q": args.jpeg_q,
    }

    for split_name, video_list in splits.items():
        if not video_list:
            continue
        out_root = output_base / split_name
        build_split_labels(
            video_list, videos_dir, annotations_dir, out_root, **frame_params
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess TEyeD video data into frames."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the raw TEyeD dataset (e.g., .../Dikablis)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to save the preprocessed frames and labels",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Path to the directory with train/val/test.txt files",
    )

    parser.add_argument("--frame_width", type=int, default=160)
    parser.add_argument("--frame_height", type=int, default=160)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--jpeg_q", type=int, default=4)

    parser.add_argument(
        "--train_sample_frac",
        type=float,
        default=1.0,
        help="Fraction of training videos to sample (e.g., 0.05 for 5%)",
    )
    parser.add_argument(
        "--val_sample_frac",
        type=float,
        default=1.0,
        help="Fraction of validation videos to sample (e.g., 0.05 for 5%)",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
