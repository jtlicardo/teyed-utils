import argparse
from pathlib import Path
import cv2


def resolve_video_path(name: str) -> Path:
    base = Path("TEyeD/Dikablis/VIDEOS")
    candidate = base / name
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".mp4")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find video at {candidate}")


def play_video(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    window = f"Playing {video_path.name} (q to quit)"
    paused = False
    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

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
    args = parser.parse_args()

    video_path = resolve_video_path(args.name)
    play_video(video_path)


if __name__ == "__main__":
    main()
