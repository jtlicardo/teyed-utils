import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract every frame from a video file"
    )
    parser.add_argument(
        "--video",
        help="Path to the input video",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path("frames") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = output_dir / f"frame_{frame_index + 1:05d}.png"
        cv2.imwrite(str(frame_filename), frame)
        frame_index += 1

    cap.release()
    print(f"Extracted {frame_index} frames into {output_dir}.")


if __name__ == "__main__":
    main()
