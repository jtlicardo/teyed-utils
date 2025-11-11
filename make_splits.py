"""
Splits all .mp4 videos in the given folder into train/val/test splits
and writes the filenames into train.txt, val.txt, and test.txt.

NOTE: Update 'videos_dir' to point to the correct videos directory.
"""

import random
from pathlib import Path

random.seed(42)

videos_dir = Path("/content/drive/MyDrive/TEyeD_unzipped/Dikablis/VIDEOS")
all_videos = sorted([v.name for v in videos_dir.glob("*.mp4")])
num_videos = len(all_videos)

random.shuffle(all_videos)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

n_train = int(num_videos * train_ratio)
n_val = int(num_videos * val_ratio)
n_test = num_videos - n_train - n_val

train_videos = all_videos[:n_train]
val_videos = all_videos[n_train : n_train + n_val]
test_videos = all_videos[n_train + n_val :]

print(f"Train: {len(train_videos)}")
print(f"Val: {len(val_videos)}")
print(f"Test: {len(test_videos)}")

with open("train.txt", "w") as f:
    f.write("\n".join(train_videos) + "\n")
with open("val.txt", "w") as f:
    f.write("\n".join(val_videos) + "\n")
with open("test.txt", "w") as f:
    f.write("\n".join(test_videos) + "\n")
