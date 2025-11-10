import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_gaze_vectors(path: Path) -> list[tuple[float, float, float] | None]:
    vectors: list[tuple[float, float, float] | None] = []
    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) > 3:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if (x, y, z) == (-1.0, -1.0, -1.0):
                    vectors.append(None)
                else:
                    vectors.append((x, y, z))
    return vectors


def infer_annotation_path(video_path: Path, annotation_arg: str | None) -> Path:
    if annotation_arg:
        return Path(annotation_arg)

    # e.g. videos/DikablisR_1_1.mp4 -> annotations/DikablisR_1_1.mp4gaze_vec.txt
    annotation_name = f"{video_path.name}gaze_vec.txt"
    annotation_path = Path("annotations") / annotation_name
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found. Looked for {annotation_path}. "
            "Pass --annotation to specify a custom file."
        )
    return annotation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side video and 3D gaze visualization."
    )
    parser.add_argument(
        "--video",
        default="videos/DikablisT_26_13.mp4",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--annotation",
        default=None,
        help="Path to the gaze annotation file. Defaults to annotations/<video_name>gaze_vec.txt",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    annotation_path = infer_annotation_path(video_path, args.annotation)
    gaze_vectors = parse_gaze_vectors(annotation_path)
    if not gaze_vectors:
        raise RuntimeError(f"No gaze data found in {annotation_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        raise RuntimeError("Failed to read video dimensions.")

    plt.ion()
    fig = plt.figure("Video + Gaze Viewer", figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.3)

    ax_frame = fig.add_subplot(gs[0, 0])
    ax_frame.set_title("Video")
    ax_frame.axis("off")
    img = ax_frame.imshow(np.zeros((height, width, 3), dtype=np.uint8))

    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_3d.set_title("Gaze Vector")
    for axis, label in zip(
        (ax_3d.set_xlabel, ax_3d.set_ylabel, ax_3d.set_zlabel), ("X", "Y", "Z")
    ):
        axis(label)
    ax_3d.set_xlim(-1, 1)
    ax_3d.set_ylim(-1, 1)
    ax_3d.set_zlim(0, 1.5)
    ax_3d.set_box_aspect([1, 1, 1])
    arrow = None
    text = ax_3d.text2D(0.05, 0.95, "", transform=ax_3d.transAxes)

    ax_xy = fig.add_subplot(gs[0, 2])
    ax_xy.set_title("Gaze XY (y inverted)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_xlim(-1, 1)
    ax_xy.set_ylim(-1, 1)
    ax_xy.set_aspect("equal")
    ax_xy.axhline(0, color="lightgray", linewidth=0.75)
    ax_xy.axvline(0, color="lightgray", linewidth=0.75)
    xy_arrow = None

    frame_index = 0
    paused = False

    def on_key(event):
        nonlocal paused
        if event.key == " ":
            paused = not paused

    cid = fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        while True:
            if paused:
                plt.pause(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame_index >= len(gaze_vectors):
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.set_data(frame_rgb)

            gaze = gaze_vectors[frame_index]
            if arrow is not None:
                arrow.remove()
                arrow = None
            if xy_arrow is not None:
                xy_arrow.remove()
                xy_arrow = None

            if gaze is not None:
                x, y, z = gaze
                arrow = ax_3d.quiver(
                    0,
                    0,
                    0,
                    x,
                    y,
                    z,
                    color="tab:red",
                    linewidth=2,
                    arrow_length_ratio=0.15,
                )
                xy_arrow = ax_xy.arrow(
                    0,
                    0,
                    x,
                    -y,  # invert Y so positive values point up on the plot
                    color="tab:blue",
                    linewidth=2,
                    head_width=0.05,
                    length_includes_head=True,
                )
                text.set_text(f"Frame {frame_index + 1}\n({x:.3f}, {y:.3f}, {z:.3f})")
            else:
                text.set_text(f"Frame {frame_index + 1}\nNo gaze data")

            fig.canvas.draw_idle()
            plt.pause(0.001)
            frame_index += 1
    finally:
        cap.release()
        plt.ioff()
        fig.canvas.mpl_disconnect(cid)

    print(f"Displayed {frame_index} frames.")
    plt.show()


if __name__ == "__main__":
    main()
