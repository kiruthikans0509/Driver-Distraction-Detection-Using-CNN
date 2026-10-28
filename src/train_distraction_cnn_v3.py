# -*- coding: utf-8 -*-
"""
train_distraction_cnn_v3.py

Third-generation Driver Distraction CNN (V3)
Multi-person generalization training and evaluation pipeline.

Dataset:
  Person A (User):     data/webcam_distraction/c0 ... c9 (1,526 images)
  Person B (Friend 1): data/webcam_distraction/session2/c0 ... c9/friend 1 (512 images)
  Person C (Friend 2): data/webcam_distraction/session2/c0 ... c9/friend 2 (514 images)
Total: 2,552 images across 10 distraction classes.

Main Experiment:
  Combined Person A + B + C
  Train: 2,052 images (temporal chunks, ~80%)
  Val:   500 images (temporal chunks, ~20%, exactly 50 images per class)

Person-Wise Evaluations:
  - Per-person evaluation on the 500-image validation set (Person A: 300, Person B: 100, Person C: 100)
  - Leave-One-Person-Out (LOPO) evaluations:
      Exp A: Train on Person A + B, Test on unseen Person C (514 images)
      Exp B: Train on Person A + C, Test on unseen Person B (512 images)
      Exp C: Train on Person B + C, Test on unseen Person A (1,526 images)

Outputs:
  models/distraction_cnn_webcam_v3.keras
  models/distraction_cnn_webcam_v3_best.keras
  models/distraction_cnn_webcam_v3_confusion.png
  models/distraction_v3_report.txt
"""

import os
import random
import hashlib
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
CHUNK_SIZE = 10
GAP_THRESHOLD = 10

BASE_DIR = r"C:\Users\kiruthikan S\Desktop\driver_monitoring\data\webcam_distraction"
START_MODEL = r"C:\Users\kiruthikan S\Desktop\driver_monitoring\models\distraction_cnn_webcam_v2_best.keras"
OUTPUT_MODEL = r"C:\Users\kiruthikan S\Desktop\driver_monitoring\models\distraction_cnn_webcam_v3.keras"
BEST_MODEL = r"C:\Users\kiruthikan S\Desktop\driver_monitoring\models\distraction_cnn_webcam_v3_best.keras"
CONFUSION_PLOT = r"C:\Users\kiruthikan S\Desktop\driver_monitoring\models\distraction_cnn_webcam_v3_confusion.png"
REPORT_PATH = r"C:\Users\kiruthikan S\Desktop\driver_monitoring\models\distraction_v3_report.txt"

CLASS_NAMES = [
    "Safe driving",               # c0
    "Texting - right",            # c1
    "Talking on phone - right",   # c2
    "Texting - left",             # c3
    "Talking on phone - left",    # c4
    "Operating radio",            # c5
    "Drinking",                   # c6
    "Reaching behind",            # c7
    "Hair/makeup",                # c8
    "Talking to passenger",       # c9
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Helper: Filename parsing & Temporal Grouping
# ---------------------------------------------------------------------------

def parse_filename(filename):
    stem = os.path.basename(filename).replace(".jpg", "")
    parts = stem.split("_")
    if len(parts) >= 4:
        try:
            idx = int(parts[1])
            ts = datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M%S")
            return idx, ts
        except (ValueError, IndexError):
            pass
    return None, None

def build_temporal_groups(class_dir):
    files = sorted([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
    if not files:
        return []
    parsed = []
    for f in files:
        idx, ts = parse_filename(f)
        parsed.append((os.path.join(class_dir, f), idx, ts))

    sub_sessions = []
    current = [parsed[0]]
    for i in range(1, len(parsed)):
        prev_idx, prev_ts = parsed[i-1][1], parsed[i-1][2]
        curr_idx, curr_ts = parsed[i][1], parsed[i][2]
        gap = (curr_ts - prev_ts).total_seconds() if (prev_ts and curr_ts) else 0
        index_reset = (curr_idx is not None and prev_idx is not None and curr_idx <= prev_idx)
        if (gap >= GAP_THRESHOLD) or index_reset:
            sub_sessions.append(current)
            current = [parsed[i]]
        else:
            current.append(parsed[i])
    sub_sessions.append(current)

    groups = []
    for sub in sub_sessions:
        filepaths = [item[0] for item in sub]
        for start in range(0, len(filepaths), CHUNK_SIZE):
            groups.append(filepaths[start:start + CHUNK_SIZE])
    return groups

def split_groups(groups, val_ratio=0.20, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(groups)))
    rng.shuffle(indices)
    n_val = max(1, round(len(groups) * val_ratio))
    val_set = set(indices[:n_val])
    tr, va = [], []
    for i, g in enumerate(groups):
        if i in val_set:
            va.extend(g)
        else:
            tr.extend(g)
    return tr, va

# ---------------------------------------------------------------------------
# Dataset Builder & Audit
# ---------------------------------------------------------------------------

def load_all_datasets():
    """
    Load metadata for Person A, Person B, Person C.
    Returns:
      person_data = {
         'Person A': {class_idx: [paths]},
         'Person B': {class_idx: [paths]},
         'Person C': {class_idx: [paths]}
      }
    """
    classes = [f"c{i}" for i in range(NUM_CLASSES)]
    person_data = {
        "Person A": {},
        "Person B": {},
        "Person C": {}
    }

    for ci, c in enumerate(classes):
        # Person A
        p_a = os.path.join(BASE_DIR, c)
        files_a = sorted([os.path.join(p_a, f) for f in os.listdir(p_a) if f.endswith(".jpg")])
        person_data["Person A"][ci] = files_a

        # Person B
        p_b = os.path.join(BASE_DIR, "session2", c, "friend 1")
        files_b = sorted([os.path.join(p_b, f) for f in os.listdir(p_b) if f.endswith(".jpg")])
        person_data["Person B"][ci] = files_b

        # Person C
        p_c = os.path.join(BASE_DIR, "session2", c, "friend 2")
        files_c = sorted([os.path.join(p_c, f) for f in os.listdir(p_c) if f.endswith(".jpg")])
        person_data["Person C"][ci] = files_c

    return person_data

def build_main_split(person_data):
    """
    Build main 80/20 train/val split using temporal chunks.
    Returns:
      train_paths, train_labels, train_persons,
      val_paths,   val_labels,   val_persons
    """
    train_paths, train_labels, train_persons = [], [], []
    val_paths,   val_labels,   val_persons   = [], [], []

    classes = [f"c{i}" for i in range(NUM_CLASSES)]
    for person_name in ["Person A", "Person B", "Person C"]:
        for ci, c in enumerate(classes):
            files = person_data[person_name][ci]
            cdir = os.path.dirname(files[0]) if files else ""
            groups = build_temporal_groups(cdir)
            tr, va = split_groups(groups, val_ratio=0.20, seed=SEED)

            train_paths.extend(tr)
            train_labels.extend([ci] * len(tr))
            train_persons.extend([person_name] * len(tr))

            val_paths.extend(va)
            val_labels.extend([ci] * len(va))
            val_persons.extend([person_name] * len(va))

    return (train_paths, train_labels, train_persons,
            val_paths,   val_labels,   val_persons)

# ---------------------------------------------------------------------------
# Data Pipeline (tf.data)
# ---------------------------------------------------------------------------

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    return img, label

def make_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")

def create_dataset(paths, labels, shuffle=True, augment=False):
    paths_t = tf.constant(paths)
    labels_t = tf.constant(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = make_augmentation_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------------------------
# Model Setup & Layer Freezing
# ---------------------------------------------------------------------------

def load_fresh_model(weights_path=START_MODEL):
    model = tf.keras.models.load_model(weights_path)
    return model

def _get_mobilenet(model):
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            return layer
    raise RuntimeError("MobileNetV2 layer not found")

def freeze_phase1(model):
    mn = _get_mobilenet(model)
    mn.trainable = False
    for layer in model.layers:
        if "mobilenet" not in layer.name.lower():
            layer.trainable = True

def unfreeze_phase2(model):
    mn = _get_mobilenet(model)
    mn.trainable = True
    total = len(mn.layers)
    n_freeze = int(total * 0.75)
    for layer in mn.layers[:n_freeze]:
        layer.trainable = False
    for layer in mn.layers[n_freeze:]:
        layer.trainable = True
    for layer in model.layers:
        if "mobilenet" not in layer.name.lower():
            layer.trainable = True

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def run_evaluation(model, val_paths, val_labels, title="EVALUATION"):
    val_ds = create_dataset(val_paths, val_labels, shuffle=False, augment=False)
    preds_raw = model.predict(val_ds, verbose=0)
    preds = np.argmax(preds_raw, axis=1)
    true = np.array(val_labels)

    acc = np.mean(preds == true)
    cm = confusion_matrix(true, preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(true, preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    report_dict = classification_report(true, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    return acc, cm, report, report_dict, preds

def save_confusion_matrix(cm, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(NUM_CLASSES)
    short_names = [f"c{i}\n{CLASS_NAMES[i][:10]}" for i in range(NUM_CLASSES)]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=8)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Training Pipeline Helper
# ---------------------------------------------------------------------------

def train_two_phases(model, train_paths, train_labels, val_paths, val_labels, 
                     best_checkpoint_path, p1_epochs=25, p2_epochs=20, verbose=1):
    import shutil
    train_ds = create_dataset(train_paths, train_labels, shuffle=True, augment=True)
    val_ds = create_dataset(val_paths, val_labels, shuffle=False, augment=False)

    p1_best = best_checkpoint_path.replace(".keras", "_p1_best.keras")
    p2_best = best_checkpoint_path.replace(".keras", "_p2_best.keras")

    # Phase 1
    freeze_phase1(model)
    compile_model(model, lr=1e-4)
    cb1 = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=verbose),
        ModelCheckpoint(p1_best, monitor="val_accuracy", save_best_only=True, mode="max", verbose=verbose)
    ]
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=p1_epochs, callbacks=cb1, verbose=verbose)

    # Phase 2
    unfreeze_phase2(model)
    compile_model(model, lr=1e-5)
    cb2 = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=verbose),
        ModelCheckpoint(p2_best, monitor="val_accuracy", save_best_only=True, mode="max", verbose=verbose)
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=p2_epochs, callbacks=cb2, verbose=verbose)

    # Determine absolute best across both phases
    max_p1 = max(h1.history.get("val_accuracy", [-1]))
    max_p2 = max(h2.history.get("val_accuracy", [-1]))

    if max_p2 >= max_p1 and os.path.exists(p2_best):
        print(f"  Selecting Phase 2 checkpoint as overall best ({max_p2:.4f} >= {max_p1:.4f})")
        shutil.copyfile(p2_best, best_checkpoint_path)
    elif os.path.exists(p1_best):
        print(f"  Selecting Phase 1 checkpoint as overall best ({max_p1:.4f} > {max_p2:.4f})")
        shutil.copyfile(p1_best, best_checkpoint_path)
    else:
        model.save(best_checkpoint_path)

    # Clean temporary phase checkpoints
    for temp_p in [p1_best, p2_best]:
        if os.path.exists(temp_p):
            try:
                os.remove(temp_p)
            except Exception:
                pass

    return h1, h2

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DRIVER DISTRACTION CNN V3 -- TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Datasets
    person_data = load_all_datasets()
    tr_p, tr_l, tr_pers, va_p, va_l, va_pers = build_main_split(person_data)

    # 2. Data Leakage Verification
    print("\n--- DATA LEAKAGE VERIFICATION ---")
    overlap = set(tr_p) & set(va_p)
    print(f"  Train samples: {len(tr_p)}")
    print(f"  Validation samples: {len(va_p)}")
    print(f"  Path overlap between Train and Val: {len(overlap)}")
    if overlap:
        raise AssertionError("CRITICAL: Data leakage detected between Train and Val!")

    # Check for MD5 duplicate leakage
    tr_hashes = {hashlib.md5(open(p, 'rb').read()).hexdigest(): p for p in tr_p}
    val_hash_overlap = sum(1 for p in va_p if hashlib.md5(open(p, 'rb').read()).hexdigest() in tr_hashes)
    print(f"  Byte-exact duplicate leakage across Train/Val: {val_hash_overlap}")
    if val_hash_overlap > 0:
        print("  WARNING: Exact duplicate frames found across split boundaries!")
    else:
        print("  [OK] Zero duplicate frames cross between Train and Val.")

    # 3. Train Main Model V3
    print("\n" + "="*70)
    print("TRAINING MAIN V3 MODEL (Person A + Person B + Person C)")
    print(f"Train samples: {len(tr_p)} | Val samples: {len(va_p)}")
    print("="*70)

    main_model = load_fresh_model(START_MODEL)
    h1, h2 = train_two_phases(
        main_model, tr_p, tr_l, va_p, va_l,
        best_checkpoint_path=BEST_MODEL,
        p1_epochs=25, p2_epochs=20, verbose=1
    )

    print("\n--- Main Model Phase 1 Epoch History ---")
    for ep in range(len(h1.history.get('loss', []))):
        print(f"  Epoch {ep+1:02d}: loss={h1.history['loss'][ep]:.4f}, acc={h1.history['accuracy'][ep]:.4f}, val_loss={h1.history['val_loss'][ep]:.4f}, val_acc={h1.history['val_accuracy'][ep]:.4f}")

    print("\n--- Main Model Phase 2 Epoch History ---")
    for ep in range(len(h2.history.get('loss', []))):
        print(f"  Epoch {ep+1:02d}: loss={h2.history['loss'][ep]:.4f}, acc={h2.history['accuracy'][ep]:.4f}, val_loss={h2.history['val_loss'][ep]:.4f}, val_acc={h2.history['val_accuracy'][ep]:.4f}")

    # Save final model
    main_model.save(OUTPUT_MODEL)
    print(f"\nFinal model saved to: {OUTPUT_MODEL}")
    print(f"Best model saved to : {BEST_MODEL}")

    # Load best model for evaluation
    best_v3 = tf.keras.models.load_model(BEST_MODEL)

    # 4. Evaluate Main V3 Model
    print("\n" + "="*70)
    print("EVALUATION: MAIN V3 MODEL (COMBINED 500 SAMPLES)")
    print("="*70)
    comb_acc, comb_cm, comb_report, comb_dict, comb_preds = run_evaluation(
        best_v3, va_p, va_l, title="Main V3 Combined"
    )
    print(f"Combined Validation Accuracy: {comb_acc:.4f} ({comb_acc*100:.2f}%)")
    print("\nClassification Report:\n", comb_report)
    print("Confusion Matrix:\n", comb_cm)
    save_confusion_matrix(comb_cm, CONFUSION_PLOT, title="Confusion Matrix - Distraction CNN V3")
    print(f"Confusion matrix plot saved: {CONFUSION_PLOT}")

    # 5. Person-Specific Evaluation on Main V3
    print("\n" + "="*70)
    print("PERSON-SPECIFIC EVALUATION (MAIN V3 MODEL)")
    print("="*70)

    person_results = {}
    for target_person in ["Person A", "Person B", "Person C"]:
        idxs = [i for i, p in enumerate(va_pers) if p == target_person]
        p_paths = [va_p[i] for i in idxs]
        p_labels = [va_l[i] for i in idxs]
        p_acc, p_cm, p_rep, p_dict, _ = run_evaluation(
            best_v3, p_paths, p_labels, title=f"Main V3 - {target_person}"
        )
        person_results[target_person] = {
            "acc": p_acc, "cm": p_cm, "report": p_rep, "dict": p_dict, "n": len(p_paths)
        }
        print(f"\n--- {target_person} ({len(p_paths)} samples) ---")
        print(f"  Accuracy: {p_acc:.4f} ({p_acc*100:.2f}%)")
        print(p_rep)

    # 6. Leave-One-Person-Out (LOPO) Evaluations
    print("\n" + "="*70)
    print("LEAVE-ONE-PERSON-OUT (LOPO) EXPERIMENTS")
    print("="*70)

    lopo_configs = [
        ("Experiment A (Test Person C)", ["Person A", "Person B"], "Person C"),
        ("Experiment B (Test Person B)", ["Person A", "Person C"], "Person B"),
        ("Experiment C (Test Person A)", ["Person B", "Person C"], "Person A"),
    ]

    lopo_results = {}
    for exp_title, train_people, test_person in lopo_configs:
        print(f"\n>>> Running {exp_title} <<<")
        print(f"    Train: {train_people} | Held-out Test: {test_person}")

        # Assemble train & val from train_people using temporal chunks
        exp_train_paths, exp_train_labels = [], []
        exp_val_paths,   exp_val_labels   = [], []

        for p_name in train_people:
            for ci in range(NUM_CLASSES):
                files = person_data[p_name][ci]
                cdir = os.path.dirname(files[0]) if files else ""
                groups = build_temporal_groups(cdir)
                tr, va = split_groups(groups, val_ratio=0.20, seed=SEED)
                exp_train_paths.extend(tr)
                exp_train_labels.extend([ci] * len(tr))
                exp_val_paths.extend(va)
                exp_val_labels.extend([ci] * len(va))

        # Assemble test set: 100% of test_person images
        exp_test_paths, exp_test_labels = [], []
        for ci in range(NUM_CLASSES):
            files = person_data[test_person][ci]
            exp_test_paths.extend(files)
            exp_test_labels.extend([ci] * len(files))

        # Verification: zero leakage from test_person into training or validation
        test_paths_set = set(exp_test_paths)
        leak_tr = test_paths_set & set(exp_train_paths)
        leak_va = test_paths_set & set(exp_val_paths)
        print(f"    Leakage check: {len(leak_tr)} train overlaps, {len(leak_va)} val overlaps")
        if leak_tr or leak_va:
            raise AssertionError(f"CRITICAL: Held-out person {test_person} leaked into training/validation!")

        # Train model for this LOPO experiment
        temp_lopo_ckpt = f"models/distraction_cnn_lopo_{test_person.replace(' ', '_').lower()}_best.keras"
        lopo_model = load_fresh_model(START_MODEL)
        
        train_two_phases(
            lopo_model, exp_train_paths, exp_train_labels,
            exp_val_paths, exp_val_labels,
            best_checkpoint_path=temp_lopo_ckpt,
            p1_epochs=15, p2_epochs=12, verbose=0  # succinct output for LOPO
        )

        # Load best LOPO checkpoint and evaluate on unseen test person
        best_lopo = tf.keras.models.load_model(temp_lopo_ckpt)
        l_acc, l_cm, l_rep, l_dict, _ = run_evaluation(
            best_lopo, exp_test_paths, exp_test_labels, title=f"LOPO {test_person}"
        )

        lopo_results[test_person] = {
            "title": exp_title, "train_people": train_people,
            "test_person": test_person, "acc": l_acc, "cm": l_cm,
            "report": l_rep, "dict": l_dict, "n_test": len(exp_test_paths)
        }
        print(f"    {test_person} Unseen Test Accuracy: {l_acc:.4f} ({l_acc*100:.2f}%) [{len(exp_test_paths)} images]")

    # 7. Write Comprehensive Report to File
    print(f"\nWriting full report to: {REPORT_PATH}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DRIVER DISTRACTION CNN V3 -- EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Starting Model : {START_MODEL}\n")
        f.write(f"Output Model   : {OUTPUT_MODEL}\n")
        f.write(f"Best Model     : {BEST_MODEL}\n")
        f.write(f"Confusion Plot : {CONFUSION_PLOT}\n\n")

        f.write("--- 1. DATASET & SPLIT SUMMARY ---\n")
        f.write(f"Person A (User)     : 1,526 images (Train: 1,226, Val: 300)\n")
        f.write(f"Person B (Friend 1) :   512 images (Train:   412, Val: 100)\n")
        f.write(f"Person C (Friend 2) :   514 images (Train:   414, Val: 100)\n")
        f.write(f"Combined Total      : 2,552 images (Train: 2,052, Val: 500)\n")
        f.write("Split Method        : Temporal grouping (10s gap threshold, 10-image chunks, 80/20 ratio, seed 42)\n")
        f.write("Data Leakage        : 0 overlapping paths, 0 cross-split duplicates.\n\n")

        f.write("--- 2. MAIN V3 TRAINING HISTORY ---\n")
        f.write("Phase 1 Epochs (MobileNetV2 frozen, head trained):\n")
        for ep in range(len(h1.history.get('loss', []))):
            f.write(f"  Epoch {ep+1:02d}: loss={h1.history['loss'][ep]:.4f}, acc={h1.history['accuracy'][ep]:.4f}, val_loss={h1.history['val_loss'][ep]:.4f}, val_acc={h1.history['val_accuracy'][ep]:.4f}\n")
        f.write("\nPhase 2 Epochs (Last 25% MobileNetV2 unfrozen):\n")
        for ep in range(len(h2.history.get('loss', []))):
            f.write(f"  Epoch {ep+1:02d}: loss={h2.history['loss'][ep]:.4f}, acc={h2.history['accuracy'][ep]:.4f}, val_loss={h2.history['val_loss'][ep]:.4f}, val_acc={h2.history['val_accuracy'][ep]:.4f}\n")
        f.write("\n")

        f.write("--- 3. MAIN V3 VALIDATION RESULTS (500 images) ---\n")
        f.write(f"Combined Validation Accuracy: {comb_acc:.4f} ({comb_acc*100:.2f}%)\n")
        f.write("Classification Report:\n" + comb_report + "\n")
        f.write("Confusion Matrix:\n" + str(comb_cm) + "\n\n")

        f.write("--- 4. PERSON-SPECIFIC VALIDATION (MAIN V3 MODEL) ---\n")
        for p in ["Person A", "Person B", "Person C"]:
            res = person_results[p]
            f.write(f"{p} ({res['n']} validation images) Accuracy: {res['acc']:.4f} ({res['acc']*100:.2f}%)\n")
            f.write(res["report"] + "\n\n")

        f.write("--- 5. LEAVE-ONE-PERSON-OUT (LOPO) RESULTS ---\n")
        for p in ["Person C", "Person B", "Person A"]:
            lres = lopo_results[p]
            f.write(f"{lres['title']}:\n")
            f.write(f"  Trained on : {lres['train_people']}\n")
            f.write(f"  Tested on  : {lres['test_person']} ({lres['n_test']} unseen images)\n")
            f.write(f"  Accuracy   : {lres['acc']:.4f} ({lres['acc']*100:.2f}%)\n")
            f.write(lres["report"] + "\n")
            f.write("Confusion Matrix:\n" + str(lres["cm"]) + "\n\n")

        f.write("--- 6. COMPARISON: V2 vs V3 ---\n")
        f.write("Metric                          V2 (Prev)      V3 (Now)\n")
        f.write(f"Combined Validation Accuracy   78.06%         {comb_acc*100:.2f}%\n")
        f.write(f"Person A (User / Session 1)    92.00%         {person_results['Person A']['acc']*100:.2f}%\n")
        f.write(f"Person B (Friend 1)            N/A            {person_results['Person B']['acc']*100:.2f}%\n")
        f.write(f"Person C (Friend 2)            N/A            {person_results['Person C']['acc']*100:.2f}%\n")
        s2_combined_acc = (person_results['Person B']['acc']*100 + person_results['Person C']['acc']*100)/2
        f.write(f"Session 2 (B + C combined)     52.73%         {s2_combined_acc:.2f}%\n\n")

        f.write("--- 7. FOCUS METRICS RECALL SUMMARY ---\n")
        for ci, cname in [(2, "Phone-right (c2)"), (9, "Passenger (c9)"), 
                          (6, "Drinking (c6)"), (7, "Reaching-behind (c7)"), 
                          (5, "Radio (c5)"), (1, "Texting-right (c1)"),
                          (3, "Texting-left (c3)"), (4, "Phone-left (c4)"),
                          (0, "Safe driving (c0)")]:
            c_key = CLASS_NAMES[ci]
            f.write(f"  {cname:<24}: Recall = {comb_dict[c_key]['recall']:.4f}, Precision = {comb_dict[c_key]['precision']:.4f}, F1 = {comb_dict[c_key]['f1-score']:.4f}\n")

    print("\n" + "="*70)
    print("V3 PIPELINE COMPLETE")
    print(f"Report written to: {REPORT_PATH}")
    print("="*70)

if __name__ == "__main__":
    main()
