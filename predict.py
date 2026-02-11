import argparse
from pathlib import Path

import cv2
import keras
import numpy as np

SEG_PARTS = ("pupil", "iris", "lid")
SEG_COLORS_BGR: dict[str, tuple[float, float, float]] = {
    "pupil": (0.0, 0.0, 255.0),  # red
    "iris": (0.0, 255.0, 0.0),  # green
    "lid": (255.0, 0.0, 0.0),  # blue
}
SEG_OVERLAY_ORDER = ("lid", "iris", "pupil")


def label_to_pixel(
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    invert_y: bool,
) -> tuple[float, float]:
    if invert_y:
        y = -y

    cx, cy = width / 2.0, height / 2.0
    px = cx + x * (width / 2.0)
    py = cy + y * (height / 2.0)
    return px, py


def prepare_frame(frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv2.resize(
        frame_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA
    )
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    arr = resized.astype(np.float32) / 255.0
    arr = (arr - 0.5) * 2.0
    return arr


def run_prediction(
    model: keras.Model, input_frame: np.ndarray
) -> tuple[tuple[float, float], dict[str, np.ndarray] | None]:
    batch = np.expand_dims(input_frame, axis=0)
    pred = model.predict(batch, verbose=0)

    gaze = None
    seg = None

    if isinstance(pred, dict):
        gaze = pred.get("gaze", None)
        if "seg" in pred:
            seg = pred.get("seg", None)
        else:
            seg = {}
            for part in SEG_PARTS:
                key = f"seg_{part}"
                if key in pred:
                    part_mask = np.asarray(pred[key]).squeeze()
                    if part_mask.ndim == 3:
                        if part_mask.shape[-1] == 1:
                            part_mask = part_mask[:, :, 0]
                        elif part_mask.shape[0] == 1:
                            part_mask = part_mask[0, :, :]
                    if part_mask.ndim != 2:
                        raise RuntimeError(
                            f"Model {key} output has unexpected shape: {part_mask.shape}"
                        )
                    seg[part] = part_mask.astype(np.float32)
            if not seg:
                seg = None
    elif isinstance(pred, (list, tuple)):
        if len(pred) >= 1:
            gaze = pred[0]
        if len(pred) >= 2:
            seg = pred[1]
    else:
        gaze = pred

    if gaze is None:
        raise RuntimeError("Model prediction did not include a gaze output.")

    gaze = np.asarray(gaze).squeeze()
    if gaze.size < 2:
        raise RuntimeError(f"Model gaze output has unexpected shape: {gaze.shape}")
    pred_xy = (float(gaze[0]), float(gaze[1]))

    if seg is None:
        return pred_xy, None

    if isinstance(seg, dict):
        return pred_xy, seg

    seg_array = np.asarray(seg).squeeze()
    if seg_array.ndim == 2:
        return pred_xy, {"pupil": seg_array.astype(np.float32)}

    if seg_array.ndim != 3:
        raise RuntimeError(f"Model seg output has unexpected shape: {seg_array.shape}")

    # Handle both HWC and CHW just in case.
    if seg_array.shape[-1] in (1, 3):
        seg_hwc = seg_array
    elif seg_array.shape[0] in (1, 3):
        seg_hwc = np.transpose(seg_array, (1, 2, 0))
    else:
        raise RuntimeError(f"Model seg output has unexpected shape: {seg_array.shape}")

    channels = seg_hwc.shape[-1]
    if channels == 1:
        return pred_xy, {"pupil": seg_hwc[:, :, 0].astype(np.float32)}
    if channels < 3:
        raise RuntimeError(
            f"Stacked segmentation output needs 1 or 3 channels, got {channels}."
        )

    seg_masks = {
        part: seg_hwc[:, :, idx].astype(np.float32)
        for idx, part in enumerate(SEG_PARTS)
    }
    return pred_xy, seg_masks


def overlay_segmentation(
    frame_bgr: np.ndarray,
    seg_masks: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
    alpha: float = 0.35,
) -> None:
    if not seg_masks:
        return

    h, w = frame_bgr.shape[:2]
    frame_f = frame_bgr.astype(np.float32)

    # Draw pupil last so it remains visible on top.
    for part in SEG_OVERLAY_ORDER:
        seg_mask = seg_masks.get(part)
        if seg_mask is None:
            continue

        seg_resized = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (seg_resized >= threshold).astype(np.float32)
        if mask.max() == 0.0:
            continue

        color = np.zeros_like(frame_bgr, dtype=np.float32)
        b, g, r = SEG_COLORS_BGR.get(part, (0.0, 0.0, 255.0))
        color[:, :, 0] = b
        color[:, :, 1] = g
        color[:, :, 2] = r

        mask3 = mask[:, :, None]
        frame_f = frame_f * (1.0 - alpha * mask3) + color * (alpha * mask3)

    np.clip(frame_f, 0.0, 255.0, out=frame_f)
    frame_bgr[:] = frame_f.astype(np.uint8)


def draw_prediction(
    frame: np.ndarray,
    pred_xy: tuple[float, float] | None,
    *,
    invert_y: bool,
    label: str | None,
) -> None:
    h, w = frame.shape[:2]
    center = (int(round(w / 2.0)), int(round(h / 2.0)))
    cv2.drawMarker(
        frame,
        center,
        (255, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(12, w // 15),
        thickness=1,
    )

    if pred_xy is None:
        if label:
            cv2.putText(
                frame,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return

    x, y = pred_xy
    px, py = label_to_pixel(x, y, w, h, invert_y=invert_y)
    target = (int(round(px)), int(round(py)))

    cv2.arrowedLine(frame, center, target, (0, 255, 0), 2, tipLength=0.15)
    cv2.drawMarker(
        frame,
        target,
        (0, 255, 0),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=max(12, w // 14),
        thickness=2,
    )

    text = f"x={x:.3f} y={y:.3f}"
    if label:
        text = f"{label} | {text}"
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gaze prediction on a video and visualize the results."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--model",
        default="trained_models/best_overall_v4.keras",
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=96,
        help="Resize frames to this square size before inference.",
    )
    parser.add_argument(
        "--invert-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert the Y sign when drawing.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated video (e.g., output.mp4).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the on-screen preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames.",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        default=0.5,
        help="Threshold for predicted segmentation overlay.",
    )
    parser.add_argument(
        "--seg-alpha",
        type=float,
        default=0.35,
        help="Overlay opacity for the segmentation mask.",
    )
    parser.add_argument(
        "--no-seg",
        action="store_true",
        help="Disable segmentation overlay (useful for gaze-only models).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    print("Loading model:", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = keras.models.load_model(model_path, compile=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            raise RuntimeError("Failed to read video dimensions.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {args.output}")

    if not args.no_display:
        cv2.namedWindow("Gaze Prediction", cv2.WINDOW_NORMAL)

    frame_index = 0
    paused = False

    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord("p"):
                    paused = False
                elif key in (ord("q"), 27):
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            input_frame = prepare_frame(frame, args.input_size)
            pred_xy, pred_seg = run_prediction(model, input_frame)

            display_frame = frame.copy()

            if (pred_seg is not None) and (not args.no_seg):
                overlay_segmentation(
                    display_frame,
                    pred_seg,
                    threshold=args.seg_threshold,
                    alpha=args.seg_alpha,
                )

            draw_prediction(
                display_frame,
                pred_xy,
                invert_y=args.invert_y,
                label=None,
            )

            if writer is not None:
                writer.write(display_frame)

            if not args.no_display:
                cv2.imshow("Gaze Prediction", display_frame)
                delay_ms = max(1, int(1000 / fps))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("p"):
                    paused = True
                elif key in (ord("q"), 27):
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames.")


if __name__ == "__main__":
    main()
