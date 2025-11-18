# TEyeD dataset util files

## Setup

1. Install [uv](https://github.com/astral-sh/uv)
2. Clone the repository:
   ```bash
   git clone https://github.com/jtlicardo/teyed-utils.git
   cd teyed-utils
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

Run scripts with `uv run <script.py> [args]`

Example:
```bash
uv run video_gaze_viewer.py
```

### Local preprocessing workflow (example)

```bash
uv run preprocess_teyed.py \
    --data_root="TEyeD/Dikablis" \
    --output_root="TEyeD_preprocessed" \
    --splits_dir="splits" \
    --frame_width=96 --frame_height=96 \
    --frame_stride=1 --jpeg_q=4 \
    --train_sample_frac=0.10 --val_sample_frac=0.00 --test_sample_frac=0.00

uv run augment_off_center.py \
    --input_root="TEyeD_preprocessed/96x96_stride5_q4_train10" \
    --output_root="TEyeD_preprocessed/96x96_stride5_q4_train10_aug" \
    --splits train --radius_threshold=0.2 --oversample_factor=3 --jpeg_quality=95
```
