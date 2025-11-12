import argparse
import os
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        "-err_detect",
        "ignore_err",
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


def chunked(items, size):
    """Yield successive chunks from the list."""
    if size <= 0:
        size = 1
    for i in range(0, len(items), size):
        yield items[i : i + size]


def stage_batch_files(batch_names, source_root: Path, stage_root: Path):
    """Copy a batch of videos/annotations to a local staging directory."""
    videos_src = source_root / "VIDEOS"
    annotations_src = source_root / "ANNOTATIONS"
    videos_stage = stage_root / "VIDEOS"
    annotations_stage = stage_root / "ANNOTATIONS"

    for directory in (videos_stage, annotations_stage):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    staged_names = []
    for name in batch_names:
        video_src = videos_src / name
        anno_src = annotations_src / f"{name}gaze_vec.txt"
        if not (video_src.exists() and anno_src.exists()):
            print(f"  Skip staging missing assets for {name}")
            continue
        shutil.copy2(video_src, videos_stage / name)
        shutil.copy2(anno_src, annotations_stage / f"{name}gaze_vec.txt")
        staged_names.append(name)

    return staged_names, videos_stage, annotations_stage


def process_video_task(
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


def process_split_parallel(
    split_name,
    video_list,
    videos_dir,
    annotations_dir,
    out_root,
    frame_width,
    frame_height,
    frame_stride,
    jpeg_q,
    num_workers,
):
    """Runs per-video processing tasks in parallel for a given split."""
    if not video_list:
        print(f"No videos for split {split_name}, skipping.")
        return []

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    print(
        f"Processing {len(video_list)} videos for split {split_name} "
        f"using {num_workers} workers..."
    )

    task_args = [
        (
            name,
            str(videos_dir),
            str(annotations_dir),
            str(out_root),
            frame_width,
            frame_height,
            frame_stride,
            jpeg_q,
        )
        for name in video_list
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_video_task, *args) for args in task_args]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = result["status"]
            name = result["name"]
            if status == "ok":
                rows.extend(result["rows"])
                print(
                    f"  [{idx}/{len(video_list)}] {name}: "
                    f"frames_saved={result['frames_saved']}, "
                    f"labels_used={result['labels_used']}"
                )
            elif status == "missing":
                print(f"  [{idx}/{len(video_list)}] Skip missing: {name}")
            elif status == "empty":
                print(f"  [{idx}/{len(video_list)}] No frames/labels for {name}")
            else:
                print(
                    f"  [{idx}/{len(video_list)}] Error processing {name}: "
                    f"{result['message']}"
                )

    if not rows:
        print(f"No rows generated for split {split_name}.")
    return rows


def main(args):
    """Main function to run the preprocessing pipeline."""
    random.seed(args.seed)

    # Define paths
    data_root = Path(args.data_root)
    stage_source_root = (
        Path(args.stage_source_root) if args.stage_source_root else None
    )
    if stage_source_root:
        if stage_source_root.resolve() == data_root.resolve():
            raise ValueError(
                "stage_source_root must differ from data_root when staging is enabled"
            )
        assert stage_source_root.exists(), f"Missing stage source root: {stage_source_root}"
        data_root.mkdir(parents=True, exist_ok=True)
        print(
            f"Batch staging enabled. Copying subsets from {stage_source_root} into {data_root}"
        )
    else:
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

    num_workers = args.num_workers or os.cpu_count() or 1
    # Process each split with optional staging batches
    for split_name, video_list in splits.items():
        out_root = output_base / split_name
        all_rows = []

        if stage_source_root:
            batch_size = args.stage_batch_size or num_workers
            for batch in chunked(video_list, batch_size):
                print(
                    f"Staging batch of {len(batch)} videos for split {split_name}..."
                )
                staged_names, staged_videos_dir, staged_annotations_dir = stage_batch_files(
                    batch, stage_source_root, data_root
                )
                if not staged_names:
                    continue
                batch_rows = process_split_parallel(
                    split_name,
                    staged_names,
                    staged_videos_dir,
                    staged_annotations_dir,
                    out_root,
                    frame_width=args.frame_width,
                    frame_height=args.frame_height,
                    frame_stride=args.frame_stride,
                    jpeg_q=args.jpeg_q,
                    num_workers=num_workers,
                )
                all_rows.extend(batch_rows)
        else:
            all_rows = process_split_parallel(
                split_name,
                video_list,
                videos_dir,
                annotations_dir,
                out_root,
                frame_width=args.frame_width,
                frame_height=args.frame_height,
                frame_stride=args.frame_stride,
                jpeg_q=args.jpeg_q,
                num_workers=num_workers,
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
        help=(
            "Path to the raw TEyeD dataset (if staging disabled) or the local "
            "staging directory (if --stage_source_root is set)."
        ),
    )
    parser.add_argument(
        "--stage_source_root",
        type=str,
        default=None,
        help=(
            "Optional path to the original dataset (e.g., Google Drive). When set, "
            "videos are copied from here to --data_root in small batches before "
            "processing."
        ),
    )
    parser.add_argument(
        "--stage_batch_size",
        type=int,
        default=0,
        help=(
            "Number of videos to stage per batch when --stage_source_root is used. "
            "Defaults to --num_workers."
        ),
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() or 2,
        help="Number of parallel worker processes to use.",
    )

    args = parser.parse_args()
    main(args)
