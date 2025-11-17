import argparse
from pathlib import Path
import cv2
import csv


def resolve_video_path(name: str) -> Path:
    base = Path("TEyeD/Dikablis/VIDEOS")
    candidate = base / name
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".mp4")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find video at {candidate}")


def infer_annotation_path(video_path: Path) -> Path | None:
    ann = Path("TEyeD/Dikablis/ANNOTATIONS") / f"{video_path.name}gaze_vec.txt"
    return ann if ann.exists() else None


def load_gaze_annotations(ann_path: Path) -> list[tuple[float, float] | None]:
    vectors: list[tuple[float, float] | None] = []
    with ann_path.open() as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            if (x, y, z) == (-1.0, -1.0, -1.0):
                vectors.append(None)
            else:
                vectors.append((x, y))
    return vectors


def play_video(video_path: Path, show_labels: bool) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    window = f"Playing {video_path.name} (q to quit)"
    paused = False
    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_vectors = None
    if show_labels:
        ann_path = infer_annotation_path(video_path)
        if ann_path:
            label_vectors = load_gaze_annotations(ann_path)
        else:
            print(f"No annotation file found for {video_path.name} in TEyeD/Dikablis/ANNOTATIONS")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # overlay frame number (1-based)
            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (10, 30),
                font,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if label_vectors is not None and 1 <= frame_idx <= len(label_vectors):
                vec = label_vectors[frame_idx - 1]
                if vec is not None:
                    x, y = vec
                    h, w = frame.shape[:2]
                    cx, cy = w // 2, h // 2
                    end = (
                        int(cx + x * (w // 2)),
                        int(cy - y * (h // 2)),  # invert Y so positive is up
                    )
                    cv2.arrowedLine(frame, (cx, cy), end, (0, 200, 0), 2, tipLength=0.3)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("r"), ord("R")):  # restart to first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx = 1
                paused = True
            elif key in (ord("["), 81):  # step backward one frame when paused
                if paused:
                    pos = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_idx = pos + 1
            elif key in (ord("]"), 83):  # step forward one frame when paused
                if paused:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a TEyeD Dikablis video by name.")
    parser.add_argument(
        "name",
        help='Video name, e.g. "DikablisSS_24_1" or "DikablisSS_24_1.mp4"',
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Overlay ground-truth gaze from TEyeD/Dikablis/ANNOTATIONS/<video>.mp4gaze_vec.txt if available.",
    )
    args = parser.parse_args()

    video_path = resolve_video_path(args.name)
    play_video(video_path, show_labels=args.show_labels)


if __name__ == "__main__":
    main()
