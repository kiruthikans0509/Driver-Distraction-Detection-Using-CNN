# -*- coding: utf-8 -*-
"""
train_distraction_cnn_v2.py

Second-generation fine-tuning of the MobileNetV2 distraction classifier.
Uses combined Session 1 + Session 2 webcam data.

Key improvements over v1:
  - Temporal burst grouping prevents near-duplicate leakage across train/val
  - Session-aware split: session origin tracked for cross-session evaluation
  - Starts from distraction_cnn_webcam_best.keras (already webcam-adapted)
  - Reports per-session validation accuracy in addition to combined metrics
  - Saves to new model files; existing models are never overwritten

Dataset layout:
  Session 1: data/webcam_distraction/c0 ... c9   (root-level)
  Session 2: data/webcam_distraction/session2/c0 ... c9

Author: Driver Monitoring System project
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend -- safe for training scripts
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
VAL_RATIO       = 0.20           # target fraction of GROUPS going to validation
CHUNK_SIZE      = 10             # images per sub-chunk when a recording sub-session
                                 # is large (prevents a single-group class having no split)
GAP_THRESHOLD   = 10            # seconds -- gap >= this starts a new recording sub-session

BASE_DIR        = "data/webcam_distraction"
SESSION1_DIR    = BASE_DIR                          # root-level
SESSION2_DIR    = os.path.join(BASE_DIR, "session2")

START_MODEL     = "models/distraction_cnn_webcam_best.keras"
OUTPUT_MODEL    = "models/distraction_cnn_webcam_v2.keras"
BEST_MODEL      = "models/distraction_cnn_webcam_v2_best.keras"
CONFUSION_PLOT  = "models/distraction_cnn_webcam_v2_confusion.png"

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
# Burst-aware grouping
# ---------------------------------------------------------------------------

def parse_filename(filename):
    """
    Parse img_NNNN_YYYYMMDD_HHMMSS.jpg
    Returns (sequential_index: int, timestamp: datetime) or (None, None).
    """
    stem = filename.replace(".jpg", "")
    parts = stem.split("_")
    if len(parts) >= 4:
        try:
            idx = int(parts[1])
            ts  = datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M%S")
            return idx, ts
        except (ValueError, IndexError):
            pass
    return None, None


def build_temporal_groups(class_dir):
    """
    Build temporal groups for images in a class directory.

    Strategy (two-level):
      1. Detect recording sub-sessions by index reset OR time gap >= GAP_THRESHOLD.
         This splits a directory into naturally separate recording moments.
      2. Within each sub-session, further divide into fixed-size chunks of CHUNK_SIZE.
         This ensures classes collected in a single long continuous recording still
         have enough groups for a meaningful 80/20 split.

    Returns a list of groups, where each group is a list of full file paths.
    """
    files = sorted([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
    if not files:
        return []

    parsed = []
    for f in files:
        idx, ts = parse_filename(f)
        parsed.append((os.path.join(class_dir, f), idx, ts))

    # --- Step 1: split into recording sub-sessions ---
    sub_sessions = []
    current = [parsed[0]]

    for i in range(1, len(parsed)):
        prev_idx, prev_ts = parsed[i-1][1], parsed[i-1][2]
        curr_idx, curr_ts = parsed[i][1],   parsed[i][2]

        gap = (curr_ts - prev_ts).total_seconds() if (prev_ts and curr_ts) else 0
        index_reset = (curr_idx is not None and prev_idx is not None
                       and curr_idx <= prev_idx)
        big_gap = gap >= GAP_THRESHOLD

        if index_reset or big_gap:
            sub_sessions.append(current)
            current = [parsed[i]]
        else:
            current.append(parsed[i])
    sub_sessions.append(current)

    # --- Step 2: chunk each sub-session into fixed-size blocks ---
    groups = []
    for sub in sub_sessions:
        filepaths = [item[0] for item in sub]
        for start in range(0, len(filepaths), CHUNK_SIZE):
            chunk = filepaths[start:start + CHUNK_SIZE]
            groups.append(chunk)

    return groups


def split_groups(groups, val_ratio=VAL_RATIO, seed=SEED):
    """
    Shuffle and split a list of groups into train/val.
    Group integrity is always preserved -- no group appears in both sets.
    Returns (train_paths, val_paths).
    """
    rng = random.Random(seed)
    indices = list(range(len(groups)))
    rng.shuffle(indices)

    n_val = max(1, round(len(groups) * val_ratio))
    val_idx_set = set(indices[:n_val])

    train_paths, val_paths = [], []
    for i, g in enumerate(groups):
        if i in val_idx_set:
            val_paths.extend(g)
        else:
            train_paths.extend(g)

    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_split():
    """
    Build combined train/val split from both sessions.
    Tracks per-session membership for cross-session evaluation.

    Returns:
        train_paths, train_labels,
        val_paths,   val_labels,
        val_session  (list of "S1"/"S2" per val image)
    """
    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []
    val_session = []  # "S1" or "S2" for each val image

    print("Building temporal groups and train/val split...")
    print(f"  GAP_THRESHOLD = {GAP_THRESHOLD}s  |  CHUNK_SIZE = {CHUNK_SIZE}  |  VAL_RATIO = {VAL_RATIO}")
    print()

    header = f"  {'Session':8} {'Class':6} {'Images':7} {'Groups':7} {'Train':7} {'Val':7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for session_label, session_dir in [("S1", SESSION1_DIR), ("S2", SESSION2_DIR)]:
        for ci in range(NUM_CLASSES):
            class_dir = os.path.join(session_dir, f"c{ci}")
            if not os.path.exists(class_dir):
                print(f"  WARNING: missing {class_dir}")
                continue

            groups = build_temporal_groups(class_dir)
            tr_p, va_p = split_groups(groups, seed=SEED)

            n_images = sum(len(g) for g in groups)
            n_groups = len(groups)

            print(f"  {session_label:8} c{ci:6} {n_images:7} {n_groups:7} {len(tr_p):7} {len(va_p):7}")

            train_paths.extend(tr_p)
            train_labels.extend([ci] * len(tr_p))

            val_paths.extend(va_p)
            val_labels.extend([ci] * len(va_p))
            val_session.extend([session_label] * len(va_p))

    print()
    print(f"  Combined train: {len(train_paths)} images")
    print(f"  Combined val  : {len(val_paths)} images")
    s1_val = sum(1 for s in val_session if s == "S1")
    s2_val = sum(1 for s in val_session if s == "S2")
    print(f"  Val breakdown  : Session 1 = {s1_val}  |  Session 2 = {s2_val}")

    return train_paths, train_labels, val_paths, val_labels, val_session


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    return img, label


def make_augmentation_layer():
    """
    Moderate augmentation.
    No horizontal flip -- left/right classes (c1/c2/c3/c4) must remain distinct.
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.06),             # +/- ~22 degrees
        layers.RandomTranslation(0.08, 0.08),    # +/- 8% shift
        layers.RandomZoom(0.10),                 # +/- 10% zoom
        layers.RandomBrightness(0.15),           # moderate brightness jitter
        layers.RandomContrast(0.15),             # moderate contrast jitter
    ], name="augmentation")


def create_dataset(paths, labels, shuffle=True, augment=False):
    paths_t  = tf.constant(paths)
    labels_t = tf.constant(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = make_augmentation_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_start_model():
    print(f"Loading starting model: {START_MODEL}")
    model = tf.keras.models.load_model(START_MODEL)
    print(f"  Total params: {model.count_params():,}")
    return model


def _get_mobilenet_layer(model):
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            return layer
    raise RuntimeError("MobileNetV2 layer not found in model")


def freeze_all_base(model):
    """Phase 1: freeze entire MobileNetV2 base, train head only."""
    mn = _get_mobilenet_layer(model)
    mn.trainable = False
    for layer in model.layers:
        if "mobilenet" not in layer.name.lower():
            layer.trainable = True

    # NOTE: Param counts from trainable_weights/non_trainable_weights may not
    # reflect the new trainable flags until after model.compile() + a forward pass.
    # We report the head-only param count (Dense layers).
    head_params = sum(
        p.numpy().size
        for layer in model.layers
        if "mobilenet" not in layer.name.lower()
        for p in layer.weights
    )
    print(f"  Phase 1 -- base frozen")
    print(f"  Head (trainable) params : {head_params:,}  "
          f"(Dense(128)+Dense(10))")
    print(f"  MobileNetV2 trainable   : {mn.trainable}  (frozen)")


def unfreeze_last_quarter(model):
    """
    Phase 2: freeze first 75% of MobileNetV2 sublayers,
    unfreeze last 25% + classification head.
    """
    mn = _get_mobilenet_layer(model)
    mn.trainable = True          # must be True for sub-layer control to work

    total = len(mn.layers)
    n_freeze = int(total * 0.75)   # freeze first 75%
    n_unfreeze = total - n_freeze  # unfreeze last 25%

    for layer in mn.layers[:n_freeze]:
        layer.trainable = False
    for layer in mn.layers[n_freeze:]:
        layer.trainable = True

    # Head always trainable
    for layer in model.layers:
        if "mobilenet" not in layer.name.lower():
            layer.trainable = True

    print(f"  Phase 2 -- partial unfreeze")
    print(f"  MobileNetV2 sublayers total : {total}")
    print(f"  Frozen (first 75%)          : {n_freeze}  ({mn.layers[0].name} -> {mn.layers[n_freeze-1].name})")
    print(f"  Unfrozen (last 25%)         : {n_unfreeze} ({mn.layers[n_freeze].name} -> {mn.layers[-1].name})")
    print(f"  MobileNetV2 trainable       : True (partial)")


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"  Compiled -- LR={lr}")


def make_callbacks(best_model_path):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, val_ds, val_labels, val_session, label="COMBINED"):
    """Full evaluation: overall, per-class, precision/recall/F1, confusion matrix."""
    print(f"\n{'='*60}")
    print(f"EVALUATION -- {label}")
    print(f"{'='*60}")

    preds_raw = model.predict(val_ds, verbose=0)
    preds     = np.argmax(preds_raw, axis=1)
    true      = np.array(val_labels)

    overall_acc = np.mean(preds == true)
    print(f"\nOverall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    print("\nPer-class accuracy:")
    for i in range(NUM_CLASSES):
        mask = true == i
        n = np.sum(mask)
        if n > 0:
            acc = np.mean(preds[mask] == true[mask])
            print(f"  c{i} {CLASS_NAMES[i]:<28s}: {acc:.4f}  ({n} samples)")

    print("\nClassification Report:")
    print(classification_report(true, preds, target_names=CLASS_NAMES, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(true, preds)
    print(cm)

    return preds, overall_acc, cm


def session_accuracy(preds, val_labels, val_session, session_tag):
    """Report accuracy for a specific session subset."""
    indices = [i for i, s in enumerate(val_session) if s == session_tag]
    if not indices:
        print(f"  {session_tag}: no samples")
        return

    true_sub = np.array([val_labels[i] for i in indices])
    pred_sub = np.array([preds[i]      for i in indices])
    acc = np.mean(pred_sub == true_sub)
    print(f"  {session_tag} accuracy: {acc:.4f} ({acc*100:.2f}%)  [{len(indices)} images]")

    for ci in range(NUM_CLASSES):
        mask = true_sub == ci
        n = np.sum(mask)
        if n > 0:
            c_acc = np.mean(pred_sub[mask] == true_sub[mask])
            print(f"    c{ci} {CLASS_NAMES[ci]:<28s}: {c_acc:.4f}  ({n})")


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(NUM_CLASSES)
    short_names = [f"c{i}\n{CLASS_NAMES[i][:10]}" for i in range(NUM_CLASSES)]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix - distraction_cnn_webcam_v2")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to: {path}")


# ---------------------------------------------------------------------------
# Sanity check (run before training)
# ---------------------------------------------------------------------------

def sanity_check(train_paths, train_labels, val_paths, val_labels, val_session):
    """
    Verify dataset integrity before any training begins.
    Prints a detailed report and raises AssertionError if a critical problem
    is found (e.g. group appears in both train and val).
    """
    print("\n" + "="*60)
    print("SANITY CHECK")
    print("="*60)

    # 1. No path overlap
    train_set = set(train_paths)
    val_set   = set(val_paths)
    overlap   = train_set & val_set
    if overlap:
        raise AssertionError(f"CRITICAL: {len(overlap)} paths appear in BOTH train and val!")
    print(f"[OK] No path overlap between train and val")

    # 2. Counts
    print(f"\nTotal train images: {len(train_paths)}")
    print(f"Total val images  : {len(val_paths)}")
    s1_val = sum(1 for s in val_session if s == "S1")
    s2_val = sum(1 for s in val_session if s == "S2")
    print(f"Val Session 1     : {s1_val}")
    print(f"Val Session 2     : {s2_val}")

    # 3. Per-class distribution
    print("\nPer-class distribution:")
    print(f"  {'Class':<32} {'Train':>7} {'Val':>7} {'S1_val':>8} {'S2_val':>8}")
    print("  " + "-"*62)
    for ci in range(NUM_CLASSES):
        tr_n  = sum(1 for l in train_labels if l == ci)
        va_n  = sum(1 for l in val_labels   if l == ci)
        s1_vn = sum(1 for i, l in enumerate(val_labels) if l == ci and val_session[i] == "S1")
        s2_vn = sum(1 for i, l in enumerate(val_labels) if l == ci and val_session[i] == "S2")
        print(f"  c{ci} {CLASS_NAMES[ci]:<28}  {tr_n:>7} {va_n:>7} {s1_vn:>8} {s2_vn:>8}")

    # 4. All val files exist on disk
    missing = [p for p in val_paths if not os.path.exists(p)]
    if missing:
        raise AssertionError(f"CRITICAL: {len(missing)} val files missing from disk")
    print(f"\n[OK] All {len(val_paths)} val files exist on disk")

    missing_tr = [p for p in train_paths if not os.path.exists(p)]
    if missing_tr:
        raise AssertionError(f"CRITICAL: {len(missing_tr)} train files missing from disk")
    print(f"[OK] All {len(train_paths)} train files exist on disk")

    print("\n[PASS] Sanity check complete")


# ---------------------------------------------------------------------------
# Layer freeze/unfreeze report (sanity check only -- no training)
# ---------------------------------------------------------------------------

def layer_report(model):
    print("\n" + "="*60)
    print("MODEL LAYER STATUS (Phase 1 -- head only)")
    print("="*60)
    mn = _get_mobilenet_layer(model)
    total = len(mn.layers)
    n_freeze = int(total * 0.75)

    print(f"  MobileNetV2 sublayers total : {total}")
    print(f"  Phase 1 base status         : Frozen ({total} sublayers)")
    print(f"  Phase 2 will freeze first   : {n_freeze} layers  ({mn.layers[0].name} .. {mn.layers[n_freeze-1].name})")
    print(f"  Phase 2 will unfreeze last  : {total-n_freeze} layers  ({mn.layers[n_freeze].name} .. {mn.layers[-1].name})")

    freeze_all_base(model)

    print(f"\nOutput models:")
    print(f"  Final  : {OUTPUT_MODEL}")
    print(f"  Best   : {BEST_MODEL}")
    print(f"  CM plot: {CONFUSION_PLOT}")

    print(f"\nAugmentation config:")
    print(f"  RandomRotation    : +/- 0.06 (fraction of 2*pi)")
    print(f"  RandomTranslation : +/- 0.08 (fraction of image)")
    print(f"  RandomZoom        : +/- 0.10")
    print(f"  RandomBrightness  : +/- 0.15")
    print(f"  RandomContrast    : +/- 0.15")
    print(f"  HorizontalFlip    : DISABLED (left/right classes c1-c4)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_ds, val_ds, phase_name, epochs):
    print(f"\n{'='*60}")
    print(f"TRAINING -- {phase_name}")
    print(f"{'='*60}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=make_callbacks(BEST_MODEL),
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DISTRACTION MODEL V2 -- FINE-TUNING")
    print("Combined Session 1 + Session 2 webcam data")
    print("=" * 60)
    print()

    # --- Build split ---
    (train_paths, train_labels,
     val_paths,   val_labels,
     val_session) = build_split()

    # --- Sanity check (always run before training) ---
    sanity_check(train_paths, train_labels, val_paths, val_labels, val_session)

    # --- Load model ---
    model = load_start_model()

    # --- Layer / model report ---
    layer_report(model)

    # ----------------------------------------------------------------
    # STOP HERE if SANITY_ONLY env var is set (used for pre-train check)
    # ----------------------------------------------------------------
    if os.environ.get("SANITY_ONLY", "0") == "1":
        print("\n[SANITY_ONLY mode] Stopping before training. Review the report above.")
        return

    # --- Create tf.data datasets ---
    print("\nCreating tf.data datasets...")
    train_ds = create_dataset(train_paths, train_labels, shuffle=True,  augment=True)
    val_ds   = create_dataset(val_paths,   val_labels,   shuffle=False, augment=False)

    # --- Phase 1: frozen base, train head ---
    print("\n" + "="*60)
    print("PHASE 1: Train classification head (MobileNetV2 frozen)")
    print("="*60)
    freeze_all_base(model)
    compile_model(model, lr=1e-4)
    history1 = train(model, train_ds, val_ds, "Phase 1 -- Head Training", epochs=30)

    print("\n--- Phase 1 Epoch History ---")
    for ep in range(len(history1.history['loss'])):
        print(f"  Epoch {ep+1:02d}: loss={history1.history['loss'][ep]:.4f}, acc={history1.history['accuracy'][ep]:.4f}, val_loss={history1.history['val_loss'][ep]:.4f}, val_acc={history1.history['val_accuracy'][ep]:.4f}")

    # --- Phase 2: partial unfreeze ---
    print("\n" + "="*60)
    print("PHASE 2: Fine-tune last 25% of MobileNetV2 + head")
    print("="*60)
    unfreeze_last_quarter(model)
    compile_model(model, lr=1e-5)
    history2 = train(model, train_ds, val_ds, "Phase 2 -- Partial Fine-Tune", epochs=30)

    print("\n--- Phase 2 Epoch History ---")
    for ep in range(len(history2.history['loss'])):
        print(f"  Epoch {ep+1:02d}: loss={history2.history['loss'][ep]:.4f}, acc={history2.history['accuracy'][ep]:.4f}, val_loss={history2.history['val_loss'][ep]:.4f}, val_acc={history2.history['val_accuracy'][ep]:.4f}")

    # --- Save final model ---
    model.save(OUTPUT_MODEL)
    print(f"\nFinal model saved: {OUTPUT_MODEL}")
    print(f"Best checkpoint  : {BEST_MODEL}")

    # --- Load best checkpoint for evaluation ---
    print(f"\nLoading best checkpoint for evaluation: {BEST_MODEL}")
    best_model = tf.keras.models.load_model(BEST_MODEL)

    # Re-create val dataset (no shuffle, no augment)
    val_ds_eval = create_dataset(val_paths, val_labels, shuffle=False, augment=False)

    # --- Combined evaluation ---
    preds, overall_acc, cm = evaluate(
        best_model, val_ds_eval, val_labels, val_session, label="COMBINED"
    )

    # --- Session-specific evaluation ---
    print("\n" + "="*60)
    print("SESSION-SPECIFIC EVALUATION")
    print("="*60)
    session_accuracy(preds, val_labels, val_session, "S1")
    print()
    session_accuracy(preds, val_labels, val_session, "S2")

    # --- Save confusion matrix ---
    save_confusion_matrix(cm, CONFUSION_PLOT)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"  Final model : {OUTPUT_MODEL}")
    print(f"  Best model  : {BEST_MODEL}")
    print(f"  CM plot     : {CONFUSION_PLOT}")
    print("="*60)


if __name__ == "__main__":
    main()
