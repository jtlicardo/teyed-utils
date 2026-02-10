"""Dataset builders for TEyeD multitask training (gaze + segmentation masks).

Designed for notebook import, e.g.:

    from training.build_dataset import (
        load_labels,
        build_train_loader_multitask,
        build_eval_loader_multitask,
        plot_train_samples,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf


SEG_PART_CHOICES = ("pupil", "iris", "lid")
DEFAULT_SEG_PARTS = ("pupil", "iris", "lid")
DEFAULT_SEG_DIMENSION = "2D"
DEFAULT_IMAGE_SHAPE = (96, 96, 3)
DEFAULT_MASK_SHAPE = (96, 96, 1)
DEFAULT_JPEG_QUALITY_OPTIONS = (35, 45, 55, 65, 75, 85, 95)
DEFAULT_OVERLAY_CMAPS = {
    "pupil": "Reds",
    "iris": "Greens",
    "lid": "Blues",
}


def load_labels(split_root: Path | str) -> pd.DataFrame:
    """Load labels.csv from a preprocessed split directory."""
    split_root = Path(split_root)
    labels_path = split_root / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    dataframe = pd.read_csv(labels_path)
    required = {"filename", "x", "y"}
    if not required.issubset(dataframe.columns):
        raise ValueError(
            "labels.csv must contain columns: filename,x,y "
            f"(found: {list(dataframe.columns)})"
        )
    return dataframe


def normalize_seg_parts(seg_parts: Sequence[str] | str) -> tuple[str, ...]:
    """Normalize/validate segmentation parts."""
    if isinstance(seg_parts, str):
        value = seg_parts.strip().lower()
        if value == "all":
            return DEFAULT_SEG_PARTS
        parts = [token.strip().lower() for token in seg_parts.split(",")]
    else:
        parts = [str(token).strip().lower() for token in seg_parts]

    normalized = []
    for part in parts:
        if not part:
            continue
        if part not in SEG_PART_CHOICES:
            raise ValueError(
                f"Invalid segmentation part '{part}'. Choose from {SEG_PART_CHOICES}."
            )
        if part not in normalized:
            normalized.append(part)

    if not normalized:
        raise ValueError(
            "No segmentation parts provided. "
            "Use e.g. seg_parts='pupil,iris,lid' or seg_parts='all'."
        )
    return tuple(normalized)


def segmentation_dirname(part: str, seg_dimension: str) -> str:
    return f"seg_{part}_{seg_dimension}"


def augment_image_photometric(image: tf.Tensor) -> tf.Tensor:
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 1.0)

    apply_gamma = tf.random.uniform([]) < 0.5
    gamma = tf.random.uniform([], 0.95, 1.10)
    image = tf.cond(
        apply_gamma,
        lambda: tf.image.adjust_gamma(image, gamma),
        lambda: image,
    )

    apply_temperature = tf.random.uniform([]) < 0.4
    delta = tf.random.uniform([], -0.06, 0.06)
    channel_gains = tf.reshape(tf.stack([1.0 - delta, 1.0, 1.0 + delta]), [1, 1, 3])
    image = tf.cond(
        apply_temperature,
        lambda: tf.clip_by_value(image * channel_gains, 0.0, 1.0),
        lambda: image,
    )

    apply_noise = tf.random.uniform([]) < 0.6
    noise_std = tf.random.uniform([], 0.0, 0.02)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=noise_std)
    image = tf.cond(
        apply_noise,
        lambda: tf.clip_by_value(image + noise, 0.0, 1.0),
        lambda: image,
    )

    blurred_3 = tf.nn.avg_pool2d(image[None], ksize=3, strides=1, padding="SAME")[0]
    apply_blur = tf.random.uniform([]) < 0.12
    image = tf.cond(apply_blur, lambda: blurred_3, lambda: image)

    apply_sharpen = tf.random.uniform([]) < 0.15
    sharpen_amount = tf.random.uniform([], 0.10, 0.25)
    sharpened = tf.clip_by_value(image + sharpen_amount * (image - blurred_3), 0.0, 1.0)
    image = tf.cond(apply_sharpen, lambda: sharpened, lambda: image)

    return image


def random_jpeg_recompression(
    image: tf.Tensor,
    *,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    quality_options: Sequence[int] = DEFAULT_JPEG_QUALITY_OPTIONS,
) -> tf.Tensor:
    image = tf.clip_by_value(image, 0.0, 1.0)
    image_uint8 = tf.cast(tf.round(image * 255.0), tf.uint8)

    quality_options = tuple(int(q) for q in quality_options)
    index = tf.random.uniform([], 0, len(quality_options), dtype=tf.int32)
    branch_fns = [
        (lambda q=q: tf.image.encode_jpeg(image_uint8, quality=q))
        for q in quality_options
    ]
    encoded = tf.switch_case(index, branch_fns=branch_fns)

    decoded = tf.image.decode_jpeg(encoded, channels=image_shape[2])
    decoded = tf.image.convert_image_dtype(decoded, tf.float32)
    decoded.set_shape(image_shape)
    return decoded


def _build_image_and_mask_paths(
    dataframe: pd.DataFrame,
    root_dir: Path | str,
    *,
    seg_parts: Sequence[str] = DEFAULT_SEG_PARTS,
    seg_dimension: str = DEFAULT_SEG_DIMENSION,
) -> tuple[list[str], dict[str, list[str]], np.ndarray]:
    root_dir = Path(root_dir)
    parts = normalize_seg_parts(seg_parts)

    image_paths = [str(root_dir / fn) for fn in dataframe["filename"].tolist()]
    mask_paths = {
        part: [
            str(root_dir / segmentation_dirname(part, seg_dimension) / Path(fn).with_suffix(".png"))
            for fn in dataframe["filename"].tolist()
        ]
        for part in parts
    }
    gazes = dataframe[["x", "y"]].astype("float32").to_numpy()
    return image_paths, mask_paths, gazes


def _decode_image_jpeg(
    path: tf.Tensor,
    *,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
) -> tf.Tensor:
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=image_shape[2])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape(image_shape)
    return image


def _decode_mask_png(
    path: tf.Tensor,
    *,
    mask_shape: tuple[int, int, int] = DEFAULT_MASK_SHAPE,
) -> tf.Tensor:
    mask = tf.image.decode_png(tf.io.read_file(path), channels=mask_shape[2])
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask.set_shape(mask_shape)
    return tf.clip_by_value(mask, 0.0, 1.0)


def _maybe_flip_multitask(
    image: tf.Tensor,
    masks: dict[str, tf.Tensor],
    gaze: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], tf.Tensor]:
    do_flip = tf.random.uniform([]) < 0.5

    image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
    flipped_masks = {
        part: tf.cond(do_flip, lambda m=mask: tf.image.flip_left_right(m), lambda m=mask: m)
        for part, mask in masks.items()
    }
    x = tf.cond(do_flip, lambda: -gaze[0], lambda: gaze[0])
    gaze = tf.stack([x, gaze[1]])
    return image, flipped_masks, gaze


def _pack_targets(
    gaze: tf.Tensor,
    masks: dict[str, tf.Tensor],
    *,
    seg_parts: Sequence[str],
    seg_target_mode: str,
) -> dict[str, tf.Tensor]:
    targets: dict[str, tf.Tensor] = {"gaze": gaze}
    parts = normalize_seg_parts(seg_parts)

    mode = seg_target_mode.strip().lower()
    if mode == "separate":
        for part in parts:
            targets[f"seg_{part}"] = masks[part]
        return targets
    if mode == "stacked":
        stacked = tf.concat([masks[part] for part in parts], axis=-1)
        targets["seg"] = stacked
        return targets
    if mode == "both":
        stacked = tf.concat([masks[part] for part in parts], axis=-1)
        targets["seg"] = stacked
        for part in parts:
            targets[f"seg_{part}"] = masks[part]
        return targets

    raise ValueError(
        "seg_target_mode must be one of: 'separate', 'stacked', 'both'. "
        f"Got: {seg_target_mode}"
    )


def build_train_loader_multitask(
    dataframe: pd.DataFrame,
    root_dir: Path | str,
    batch_size: int,
    *,
    seg_parts: Sequence[str] | str = DEFAULT_SEG_PARTS,
    seg_dimension: str = DEFAULT_SEG_DIMENSION,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    mask_shape: tuple[int, int, int] = DEFAULT_MASK_SHAPE,
    shuffle_buffer: int = 50000,
    seed: int = 42,
    seg_target_mode: str = "separate",
) -> tf.data.Dataset:
    """Build training dataset with augmentation.

    Output: (image, targets_dict) where targets_dict always includes "gaze"
    and segmentation targets based on seg_target_mode.
    """
    parts = normalize_seg_parts(seg_parts)
    image_paths, mask_paths, gazes = _build_image_and_mask_paths(
        dataframe,
        root_dir,
        seg_parts=parts,
        seg_dimension=seg_dimension,
    )

    slices = {"image_path": image_paths, "gaze": gazes}
    for part in parts:
        slices[f"mask_path_{part}"] = mask_paths[part]

    dataset = tf.data.Dataset.from_tensor_slices(slices)
    dataset = dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    def load(sample: dict[str, tf.Tensor]) -> tuple[tf.Tensor, dict[str, tf.Tensor], tf.Tensor]:
        image = _decode_image_jpeg(sample["image_path"], image_shape=image_shape)
        masks = {
            part: _decode_mask_png(sample[f"mask_path_{part}"], mask_shape=mask_shape)
            for part in parts
        }
        gaze = tf.cast(sample["gaze"], tf.float32)
        return image, masks, gaze

    def augment_and_pack(
        image: tf.Tensor,
        masks: dict[str, tf.Tensor],
        gaze: tf.Tensor,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        image = random_jpeg_recompression(image, image_shape=image_shape)
        image = augment_image_photometric(image)
        image, masks, gaze = _maybe_flip_multitask(image, masks, gaze)
        image = (image - 0.5) * 2.0
        targets = _pack_targets(
            gaze,
            masks,
            seg_parts=parts,
            seg_target_mode=seg_target_mode,
        )
        return image, targets

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment_and_pack, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_eval_loader_multitask(
    dataframe: pd.DataFrame,
    root_dir: Path | str,
    batch_size: int,
    *,
    seg_parts: Sequence[str] | str = DEFAULT_SEG_PARTS,
    seg_dimension: str = DEFAULT_SEG_DIMENSION,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    mask_shape: tuple[int, int, int] = DEFAULT_MASK_SHAPE,
    seg_target_mode: str = "separate",
    cache: bool = True,
) -> tf.data.Dataset:
    """Build eval dataset without augmentation."""
    parts = normalize_seg_parts(seg_parts)
    image_paths, mask_paths, gazes = _build_image_and_mask_paths(
        dataframe,
        root_dir,
        seg_parts=parts,
        seg_dimension=seg_dimension,
    )

    slices = {"image_path": image_paths, "gaze": gazes}
    for part in parts:
        slices[f"mask_path_{part}"] = mask_paths[part]

    dataset = tf.data.Dataset.from_tensor_slices(slices)

    def load_and_pack(sample: dict[str, tf.Tensor]) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        image = _decode_image_jpeg(sample["image_path"], image_shape=image_shape)
        masks = {
            part: _decode_mask_png(sample[f"mask_path_{part}"], mask_shape=mask_shape)
            for part in parts
        }
        gaze = tf.cast(sample["gaze"], tf.float32)
        image = (image - 0.5) * 2.0
        targets = _pack_targets(
            gaze,
            masks,
            seg_parts=parts,
            seg_target_mode=seg_target_mode,
        )
        return image, targets

    dataset = dataset.map(load_and_pack, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)


def denorm_image(image_batch: np.ndarray) -> np.ndarray:
    """Convert images from [-1, 1] to [0, 1] for visualization."""
    return np.clip((image_batch / 2.0) + 0.5, 0.0, 1.0)


def gaze_to_pixel(gaze_xy: Sequence[float], width: int, height: int) -> tuple[float, float]:
    """Map gaze offsets in [-0.5, 0.5] to image pixel coordinates."""
    x, y = float(gaze_xy[0]), float(gaze_xy[1])
    px = (x + 0.5) * float(width)
    py = (y + 0.5) * float(height)
    return px, py


def _extract_masks_from_targets(
    targets: dict[str, tf.Tensor],
    *,
    seg_parts: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if "gaze" not in targets:
        raise KeyError("targets is missing 'gaze'")

    parts = normalize_seg_parts(seg_parts)
    if "seg" in targets:
        stacked = targets["seg"].numpy()
        if stacked.ndim != 4 or stacked.shape[-1] < len(parts):
            raise ValueError(
                "targets['seg'] must have shape [B,H,W,C] with C >= len(seg_parts)."
            )
        masks = {
            part: stacked[:, :, :, index : index + 1]
            for index, part in enumerate(parts)
        }
    else:
        masks = {}
        for part in parts:
            key = f"seg_{part}"
            if key not in targets:
                raise KeyError(
                    f"targets is missing '{key}'. "
                    "Either use seg_target_mode='separate'/'both' or pass seg_target_mode='stacked' and provide seg_parts."
                )
            masks[part] = targets[key].numpy()
    gaze = targets["gaze"].numpy()
    return gaze, masks


def plot_train_samples(
    dataset: tf.data.Dataset,
    *,
    seg_parts: Sequence[str] | str = DEFAULT_SEG_PARTS,
    count: int = 9,
    seed: int = 0,
    threshold: float = 0.5,
) -> None:
    """Visualize a batch with gaze and multiple mask overlays."""
    import matplotlib.pyplot as plt

    tf.random.set_seed(seed)
    np.random.seed(seed)

    images, targets = next(iter(dataset))
    if not isinstance(targets, dict):
        raise TypeError(
            "Expected dataset targets to be a dict. "
            "Use build_*_loader_multitask from this module."
        )

    parts = normalize_seg_parts(seg_parts)
    gaze_np, masks_np = _extract_masks_from_targets(targets, seg_parts=parts)
    images_np = denorm_image(images.numpy())

    batch_size = images_np.shape[0]
    n = min(count, batch_size)
    cols = 3
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(5 * cols, 5 * rows))

    for index in range(n):
        ax = plt.subplot(rows, cols, index + 1)
        image = images_np[index]
        height, width = image.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0

        ax.imshow(image)
        # Draw pupil last so it remains visible above iris/lid overlays.
        overlay_parts = [part for part in parts if part != "pupil"]
        if "pupil" in parts:
            overlay_parts.append("pupil")

        for part in overlay_parts:
            mask = masks_np[part][index, :, :, 0]
            mask_bin = (mask >= threshold).astype(np.float32)
            ax.imshow(
                mask_bin,
                cmap=DEFAULT_OVERLAY_CMAPS.get(part, "gray"),
                alpha=mask_bin * 0.45,
                vmin=0.0,
                vmax=1.0,
            )

        gaze = gaze_np[index]
        px, py = gaze_to_pixel(gaze, width, height)

        ax.axhline(y=center_y, color="cyan", linestyle="--", alpha=0.25)
        ax.axvline(x=center_x, color="cyan", linestyle="--", alpha=0.25)
        ax.scatter([center_x], [center_y], c="cyan", s=40, marker="o")
        ax.scatter([px], [py], c="lime", s=120, marker="+", linewidths=2)

        ratios = []
        for part in parts:
            ratio = float(masks_np[part][index, :, :, 0].mean())
            ratios.append(f"{part}:{ratio:.3f}")
        text = (
            f"gaze: ({gaze[0]:+.3f}, {gaze[1]:+.3f})\n"
            f"pixel: ({px:.1f}, {py:.1f})\n"
            f"mean: {' | '.join(ratios)}"
        )
        ax.text(
            4,
            18,
            text,
            color="white",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )
        ax.set_title(f"Sample {index}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
