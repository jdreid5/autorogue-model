# split-dataset.py
import os, shutil, random
from collections import defaultdict
random.seed(13)

SRC = "data/cropped-images"
OUT = "data/split-images"
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = ["healthy-russets", "leaf-roll-russets"]

os.makedirs(OUT, exist_ok=True)
for s in SPLITS:
	for c in CLASSES: os.makedirs(os.path.join(OUT, s, c), exist_ok=True)

def group_key(fname):
	return fname

for cls in CLASSES:
	src_dir = os.path.join(SRC, cls)
	files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
	buckets = defaultdict(list)
	for f in files: buckets[group_key(f)].append(f)

	groups = list(buckets.values())
	random.shuffle(groups)
	n = len(groups)
	n_train = int(n * SPLITS["train"]); n_val = int(n * SPLITS["val"])
	split_groups = {
		"train": groups[:n_train],
		"val": groups[n_train:n_train+n_val],
		"test": groups[n_train+n_val:]
	}

	for split, glist in split_groups.items():
		for g in glist:
			for f in g:
				shutil.copy2(os.path.join(src_dir, f), os.path.join(OUT, split, cls, f))

print("Done -> data/train, data/val, data/test")