import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras


def load_frames_batched(cap, batch_size, target_size):
    frames, originals = [], []
    while len(frames) < batch_size:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        originals.append(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame_resized.astype(np.float32) / 255.0)
    if not frames:
        return None, None
    return np.stack(frames, axis=0), originals


def draw_vectors(frames_bgr, predictions, arrow_length=45, color=(0, 0, 255)):
    for i, frame in enumerate(frames_bgr):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        x, y = float(predictions[i, 0]), float(predictions[i, 1])
        end = (int(cx + x * arrow_length), int(cy + y * arrow_length))
        cv2.arrowedLine(frame, (cx, cy), end, color, 1, tipLength=0.3)


def parse_gaze_vectors(path: Path):
    """Read TEyeD gaze annotations returning list of (x, y, z) or None per frame."""
    vectors = []
    with path.open("r") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 4:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if (x, y, z) == (-1.0, -1.0, -1.0):
                vectors.append(None)
            else:
                vectors.append((x, y, z))
    return vectors


def infer_annotation_path(video_path: Path, annotation_arg: str | None) -> Path | None:
    if annotation_arg == "":
        return None
    if annotation_arg:
        return Path(annotation_arg)
    ann_name = f"{video_path.name}gaze_vec.txt"
    ann_path = Path("TEyeD/Dikablis/ANNOTATIONS") / ann_name
    return ann_path if ann_path.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input-size", type=int, default=96)
    p.add_argument("--arrow-length", type=int, default=150)
    p.add_argument(
        "--annotation",
        default=None,
        help=(
            "Optional path to TEyeD gaze annotation file; defaults to "
            "TEyeD/Dikablis/ANNOTATIONS/<video_name>gaze_vec.txt"
        ),
    )
    args = p.parse_args()

    model = keras.models.load_model(args.model)
    video_path = Path(args.input)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    annotation_path = infer_annotation_path(video_path, args.annotation)
    gt_vectors = None
    if annotation_path:
        gt_vectors = parse_gaze_vectors(annotation_path)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, src_fps, (width, height))
        print("Legend: red = prediction, green = ground truth")

    total_frames = 0
    t0 = time.perf_counter()
    target_size = (args.input_size, args.input_size)

    while True:
        batch, originals = load_frames_batched(cap, args.batch_size, target_size)
        if batch is None:
            break
        preds = model.predict(batch, verbose=0)
        if gt_vectors is not None:
            start_idx = total_frames
            for i, frame in enumerate(originals):
                gt_idx = start_idx + i
                if gt_idx >= len(gt_vectors):
                    break
                gt = gt_vectors[gt_idx]
                if gt is None:
                    continue
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                gx, gy = float(gt[0]), float(gt[1])
                end = (
                    int(cx + gx * args.arrow_length),
                    int(cy + gy * args.arrow_length),
                )
                cv2.arrowedLine(frame, (cx, cy), end, (0, 200, 0), 1, tipLength=0.3)
        frames_bgr = originals
        draw_vectors(
            frames_bgr, preds, arrow_length=args.arrow_length, color=(0, 0, 255)
        )
        for f in frames_bgr:
            if writer:
                writer.write(f)
            else:
                cv2.imshow("pred", f)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    if writer:
                        writer.release()
                    cv2.destroyAllWindows()
                    return
        total_frames += len(frames_bgr)

    t1 = time.perf_counter()
    fps = total_frames / max(1e-6, (t1 - t0))
    print(f"Processed {total_frames} frames at {fps:.1f} FPS")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
