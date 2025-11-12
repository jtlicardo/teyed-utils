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

### Local preprocessing workflow

```bash
uv run preprocess_teyed.py \
    --data_root="TEyeD/Dikablis" \
    --output_root="TEyeD_preprocessed" \
    --splits_dir="splits" \
    --frame_width=128 --frame_height=128 \
    --frame_stride=5 --jpeg_q=4 \
    --train_sample_frac=0.05 --val_sample_frac=0.05 --test_sample_frac=0.05
```
