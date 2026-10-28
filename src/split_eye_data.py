import os
import shutil
import random

random.seed(42)  # ensures reproducible splits every time you run this

SOURCE_DIR = "data/eyes"
DEST_DIR = "data/eyes_split"
CLASSES = ["Open_Eyes", "Closed_Eyes"]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

for split in SPLIT_RATIOS:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

for cls in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, split_images in splits.items():
        for img_name in split_images:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(DEST_DIR, split, cls, img_name)
            shutil.copy2(src, dst)

    print(f"{cls}: {n} total -> train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

print("Done. Split data saved to data/eyes_split/")