import argparse
import random
import subprocess
from pathlib import Path

import pandas as pd


def extract_video_ffmpeg(
    video_path: Path,
    out_dir: Path,
    size: tuple,
    stride: int,
    q: int,
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
        "-err_detect",
        "ignore_err",
    ]
    cmd.extend(
        [
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
    )
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"    ffmpeg reported errors for {video_path.name} "
            f"(return code {result.returncode}). Proceeding with any frames decoded "
            "before the failure."
        )
        if result.stderr:
            last_lines = [line for line in result.stderr.strip().splitlines() if line]
            if last_lines:
                print(f"    Last ffmpeg message: {last_lines[-1]}")
    return len(list(out_dir.glob("*.jpg")))


def process_video(
    name,
    videos_dir,
    annotations_dir,
    out_root,
    frame_width,
    frame_height,
    frame_stride,
    jpeg_q,
):
    """Processes a single video and returns metadata/rows."""
    videos_dir = Path(videos_dir)
    annotations_dir = Path(annotations_dir)
    out_root = Path(out_root)

    vpath = videos_dir / name
    apath = annotations_dir / f"{name}gaze_vec.txt"
    if not (vpath.exists() and apath.exists()):
        return {
            "name": name,
            "rows": [],
            "frames_saved": 0,
            "labels_used": 0,
            "status": "missing",
            "message": "missing video or annotation",
        }

    try:
        out_dir = out_root / vpath.stem
        saved = extract_video_ffmpeg(
            vpath,
            out_dir,
            size=(frame_width, frame_height),
            stride=frame_stride,
            q=jpeg_q,
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
            df = df.iloc[::frame_stride]  # select every n-th row

        n = min(saved, len(df))
        if n == 0:
            return {
                "name": name,
                "rows": [],
                "frames_saved": saved,
                "labels_used": n,
                "status": "empty",
                "message": "no frames/labels after extraction",
            }

        df = df.iloc[:n]
        rows = [
            (f"{vpath.stem}/{j:06d}.jpg", x, y)
            for j, (x, y) in enumerate(zip(df["x"], df["y"]), start=1)
        ]
        return {
            "name": name,
            "rows": rows,
            "frames_saved": saved,
            "labels_used": n,
            "status": "ok",
            "message": "",
        }
    except Exception as exc:
        return {
            "name": name,
            "rows": [],
            "frames_saved": 0,
            "labels_used": 0,
            "status": "error",
            "message": str(exc),
        }


def process_split(
    split_name,
    video_list,
    videos_dir,
    annotations_dir,
    out_root,
    frame_width,
    frame_height,
    frame_stride,
    jpeg_q,
):
    """Runs per-video processing sequentially for a given split."""
    if not video_list:
        print(f"No videos for split {split_name}, skipping.")
        return []

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"Processing {len(video_list)} videos for split {split_name}...")

    for idx, name in enumerate(video_list, start=1):
        result = process_video(
            name,
            videos_dir,
            annotations_dir,
            out_root,
            frame_width,
            frame_height,
            frame_stride,
            jpeg_q,
        )
        status = result["status"]
        if status == "ok":
            rows.extend(result["rows"])
            print(
                f"  [{idx}/{len(video_list)}] {name}: "
                f"frames_saved={result['frames_saved']}, labels_used={result['labels_used']}"
            )
        elif status == "missing":
            print(f"  [{idx}/{len(video_list)}] Skip missing: {name}")
        elif status == "empty":
            print(f"  [{idx}/{len(video_list)}] No frames/labels for {name}")
        else:
            print(
                f"  [{idx}/{len(video_list)}] Error processing {name}: {result['message']}"
            )

    if not rows:
        print(f"No rows generated for split {split_name}.")
    return rows


def main(args):
    """Main function to run the preprocessing pipeline."""
    random.seed(args.seed)

    # Define paths
    data_root = Path(args.data_root)
    assert data_root.exists(), f"Missing raw data root: {data_root}"

    videos_dir = data_root / "VIDEOS"
    annotations_dir = data_root / "ANNOTATIONS"

    output_parent = Path(args.output_root)
    sample_suffix = ""
    if args.train_sample_frac < 1.0:
        train_pct = int(args.train_sample_frac * 100)
        sample_suffix += f"_train{train_pct:02d}"
    if args.val_sample_frac < 1.0:
        val_pct = int(args.val_sample_frac * 100)
        sample_suffix += f"_val{val_pct:02d}"
    params_name = f"{args.frame_width}x{args.frame_height}_stride{args.frame_stride}_q{args.jpeg_q}{sample_suffix}"
    output_base = output_parent / params_name
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Saving preprocessed data to: {output_base}")

    repo_dir = Path(args.splits_dir)
    split_file_order = ["train", "val", "test"]
    splits = {}
    for split in split_file_order:
        file_path = repo_dir / f"{split}.txt"
        if not file_path.exists():
            print(f"Warning: missing split file {file_path}, skipping.")
            continue
        with open(file_path) as f:
            splits[split] = [line.strip() for line in f if line.strip()]

    sample_fracs = {
        "train": args.train_sample_frac,
        "val": args.val_sample_frac,
        "test": args.test_sample_frac,
    }
    for split_name, videos in list(splits.items()):
        frac = sample_fracs.get(split_name, 1.0)
        if frac <= 0.0:
            print(f"Skipping split {split_name} (sample fraction <= 0).")
            splits.pop(split_name)
            continue
        if frac < 1.0:
            num_to_sample = max(1, int(len(videos) * frac))
            splits[split_name] = random.sample(videos, num_to_sample)
            print(f"Sampled {len(splits[split_name])} videos for {split_name}.")

    # Process each split sequentially
    for split_name, video_list in splits.items():
        out_root = output_base / split_name
        all_rows = process_split(
            split_name,
            video_list,
            videos_dir,
            annotations_dir,
            out_root,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            frame_stride=args.frame_stride,
            jpeg_q=args.jpeg_q,
        )

        if all_rows:
            out_root.mkdir(parents=True, exist_ok=True)
            out_csv = out_root / "labels.csv"
            pd.DataFrame(all_rows, columns=["filename", "x", "y"]).to_csv(
                out_csv, index=False
            )
            print(f"Wrote {out_csv} with {len(all_rows)} rows\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess TEyeD video data into frames."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the raw TEyeD dataset (e.g., TEyeD/Dikablis)",
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
        help="Fraction of training videos to sample (e.g., 0.05 for 5%%)",
    )
    parser.add_argument(
        "--val_sample_frac",
        type=float,
        default=1.0,
        help="Fraction of validation videos to sample (e.g., 0.05 for 5%%)",
    )
    parser.add_argument(
        "--test_sample_frac",
        type=float,
        default=1.0,
        help="Fraction of test videos to sample (set 0 to skip)",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
