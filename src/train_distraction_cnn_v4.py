# -*- coding: utf-8 -*-
"""
train_distraction_cnn_v4.py

Fourth-generation Driver Distraction CNN (V4)
Multi-person generalization training with expanded 6-person dataset.

Dataset:
  6 people total across session2 and session3_raw:
  - me: 2,607 images
  - friend 1: 512 images
  - friend 2: 514 images
  - friend 3: 1,024 images
  - friend 4: 993 images
  - friend 5: 971 images (held out for unseen-person test)

Total: 6,621 images across 10 distraction classes.

V4 Level 2 Split (Unseen-Driver Benchmark):
  Train: 4,578 images (me, friend 1, friend 2, friend 3, friend 4)
  Validation: 1,072 images (me, friend 1, friend 2, friend 3, friend 4)
  Test: 971 images (friend 5 only - completely held out)

Data Source:
  CSV manifests in data/v4_splits/:
  - train.csv
  - validation.csv
  - test_unseen_person.csv

Outputs:
  models/distraction_cnn_webcam_v4.keras
  models/distraction_cnn_webcam_v4_best.keras
  models/distraction_cnn_webcam_v4_confusion.png
  models/distraction_v4_report.txt
  models/distraction_v4_metrics.json
  models/distraction_v4_history.csv
  models/distraction_v4_training.log
"""

import os
import sys
import io
import csv
import random
import json
import time
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

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

BASE_DIR = r"C:\Users\kiruthikan S\Desktop\driver_monitoring"
START_MODEL = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v3_best.keras")
OUTPUT_MODEL = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v4.keras")
BEST_MODEL = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v4_best.keras")
CONFUSION_PLOT = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v4_confusion.png")
REPORT_PATH = os.path.join(BASE_DIR, "models", "distraction_v4_report.txt")
METRICS_PATH = os.path.join(BASE_DIR, "models", "distraction_v4_metrics.json")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "distraction_v4_history.csv")
LOG_PATH = os.path.join(BASE_DIR, "models", "distraction_v4_training.log")

TRAIN_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "validation.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "test_unseen_person.csv")

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
# Logging Setup
# ---------------------------------------------------------------------------

def log_message(msg):
    """Write message to both stdout and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

# ---------------------------------------------------------------------------
# CSV Manifest Loading
# ---------------------------------------------------------------------------

def load_manifest(csv_path):
    """Load image paths and labels from CSV manifest."""
    paths = []
    labels = []
    persons = []
    temporal_groups = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["image_path"])
            labels.append(int(row["class_id"]))
            persons.append(row["person"])
            temporal_groups.append(row["temporal_group"])

    return paths, labels, persons, temporal_groups

# ---------------------------------------------------------------------------
# Data Pipeline (tf.data)
# ---------------------------------------------------------------------------

def load_and_preprocess(path, label):
    """Load and preprocess a single image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    return img, label

def make_augmentation_layer():
    """Create data augmentation layer."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")

def create_dataset(paths, labels, shuffle=True, augment=False):
    """Create tf.data Dataset from paths and labels."""
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
    """Load model from checkpoint."""
    model = tf.keras.models.load_model(weights_path)
    return model

def _get_mobilenet(model):
    """Find MobileNetV2 layer in model."""
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            return layer
    raise RuntimeError("MobileNetV2 layer not found")

def freeze_phase1(model):
    """Freeze MobileNetV2 backbone for Phase 1."""
    mn = _get_mobilenet(model)
    mn.trainable = False
    for layer in model.layers:
        if "mobilenet" not in layer.name.lower():
            layer.trainable = True

def unfreeze_phase2(model):
    """Unfreeze last 25% of MobileNetV2 for Phase 2."""
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
    """Compile model with optimizer and loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def run_evaluation(model, val_paths, val_labels, title="EVALUATION"):
    """Run evaluation on dataset."""
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
    """Save confusion matrix as PNG."""
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

def save_history_csv(history, path):
    """Save training history to CSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,accuracy,val_loss,val_accuracy,phase\n")
        for i, (loss, acc, val_loss, val_acc) in enumerate(zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"]
        )):
            f.write(f"{i+1},{loss:.6f},{acc:.6f},{val_loss:.6f},{val_acc:.6f},phase1\n")

# ---------------------------------------------------------------------------
# Training Pipeline Helper
# ---------------------------------------------------------------------------

def train_two_phases(model, train_paths, train_labels, val_paths, val_labels,
                     best_checkpoint_path, p1_epochs=15, p2_epochs=15, verbose=1):
    """Train model in two phases."""
    import shutil
    train_ds = create_dataset(train_paths, train_labels, shuffle=True, augment=True)
    val_ds = create_dataset(val_paths, val_labels, shuffle=False, augment=False)

    p1_best = best_checkpoint_path.replace(".keras", "_p1_best.keras")
    p2_best = best_checkpoint_path.replace(".keras", "_p2_best.keras")

    # Phase 1
    log_message("=" * 70)
    log_message("PHASE 1: Training classification head (MobileNetV2 frozen)")
    log_message("=" * 70)
    freeze_phase1(model)
    compile_model(model, lr=1e-4)
    cb1 = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=verbose),
        ModelCheckpoint(p1_best, monitor="val_accuracy", save_best_only=True, mode="max", verbose=verbose)
    ]
    p1_start = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=p1_epochs, callbacks=cb1, verbose=verbose)
    p1_duration = time.time() - p1_start
    log_message(f"Phase 1 completed in {p1_duration:.1f} seconds ({p1_duration/60:.1f} minutes)")

    # Phase 2
    log_message("=" * 70)
    log_message("PHASE 2: Fine-tuning last 25% of MobileNetV2")
    log_message("=" * 70)
    unfreeze_phase2(model)
    compile_model(model, lr=1e-5)
    cb2 = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=verbose),
        ModelCheckpoint(p2_best, monitor="val_accuracy", save_best_only=True, mode="max", verbose=verbose)
    ]
    p2_start = time.time()
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=p2_epochs, callbacks=cb2, verbose=verbose)
    p2_duration = time.time() - p2_start
    log_message(f"Phase 2 completed in {p2_duration:.1f} seconds ({p2_duration/60:.1f} minutes)")

    # Determine absolute best across both phases
    max_p1 = max(h1.history.get("val_accuracy", [-1]))
    max_p2 = max(h2.history.get("val_accuracy", [-1]))

    if max_p2 >= max_p1 and os.path.exists(p2_best):
        log_message(f"Selecting Phase 2 checkpoint as overall best ({max_p2:.4f} >= {max_p1:.4f})")
        shutil.copyfile(p2_best, best_checkpoint_path)
    elif os.path.exists(p1_best):
        log_message(f"Selecting Phase 1 checkpoint as overall best ({max_p1:.4f} > {max_p2:.4f})")
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

    # Save combined history
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        f.write("epoch,loss,accuracy,val_loss,val_accuracy,phase\n")
        for i, (loss, acc, val_loss, val_acc) in enumerate(zip(
            h1.history["loss"], h1.history["accuracy"],
            h1.history["val_loss"], h1.history["val_accuracy"]
        )):
            f.write(f"{i+1},{loss:.6f},{acc:.6f},{val_loss:.6f},{val_acc:.6f},phase1\n")
        for i, (loss, acc, val_loss, val_acc) in enumerate(zip(
            h2.history["loss"], h2.history["accuracy"],
            h2.history["val_loss"], h2.history["val_accuracy"]
        )):
            f.write(f"{i+len(h1.history['loss'])+1},{loss:.6f},{acc:.6f},{val_loss:.6f},{val_acc:.6f},phase2\n")

    return h1, h2, p1_duration, p2_duration

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    # Initialize log file
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"V4 Training Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    log_message("=" * 70)
    log_message("DRIVER DISTRACTION CNN V4 -- TRAINING & EVALUATION PIPELINE")
    log_message("=" * 70)

    # 1. Load Manifests
    log_message("\n[1] Loading V4 manifests...")
    train_paths, train_labels, train_persons, train_groups = load_manifest(TRAIN_CSV)
    val_paths, val_labels, val_persons, val_groups = load_manifest(VAL_CSV)
    test_paths, test_labels, test_persons, test_groups = load_manifest(TEST_CSV)

    log_message(f"  Train samples: {len(train_paths)}")
    log_message(f"  Validation samples: {len(val_paths)}")
    log_message(f"  Test samples: {len(test_paths)}")

    # 2. Data Leakage Verification
    log_message("\n[2] Data leakage verification...")
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    log_message(f"  Train/Val path overlap: {len(train_val_overlap)}")
    log_message(f"  Train/Test path overlap: {len(train_test_overlap)}")
    log_message(f"  Val/Test path overlap: {len(val_test_overlap)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        log_message("[CRITICAL ERROR] Data leakage detected!")
        exit(1)

    # Verify Friend 5 isolation
    log_message(f"  Train persons: {set(train_persons)}")
    log_message(f"  Validation persons: {set(val_persons)}")
    log_message(f"  Test persons: {set(test_persons)}")

    if "friend 5" in train_persons or "friend 5" in val_persons:
        log_message("[CRITICAL ERROR] Friend 5 found in training or validation!")
        exit(1)

    if set(test_persons) != {"friend 5"}:
        log_message("[CRITICAL ERROR] Test set contains persons other than friend 5!")
        exit(1)

    log_message("[OK] No data leakage detected")

    # 3. Train V4 Model
    log_message("\n" + "=" * 70)
    log_message("TRAINING V4 MODEL")
    log_message(f"Train samples: {len(train_paths)} | Val samples: {len(val_paths)}")
    log_message(f"Batch size: {BATCH_SIZE}")
    log_message(f"Seed: {SEED}")
    log_message("=" * 70)

    main_model = load_fresh_model(START_MODEL)
    log_message(f"Loaded starting model from: {START_MODEL}")
    log_message(f"Model parameters: {main_model.count_params():,}")

    overall_start = time.time()
    h1, h2, p1_duration, p2_duration = train_two_phases(
        main_model, train_paths, train_labels, val_paths, val_labels,
        best_checkpoint_path=BEST_MODEL,
        p1_epochs=15, p2_epochs=15, verbose=1
    )
    overall_duration = time.time() - overall_start

    log_message(f"\nTotal training time: {overall_duration:.1f} seconds ({overall_duration/60:.1f} minutes)")

    # Save final model
    main_model.save(OUTPUT_MODEL)
    log_message(f"Final model saved to: {OUTPUT_MODEL}")
    log_message(f"Best model saved to: {BEST_MODEL}")

    # Load best model for evaluation
    best_v4 = tf.keras.models.load_model(BEST_MODEL)

    # 4. Evaluate on Validation Set
    log_message("\n" + "=" * 70)
    log_message("EVALUATION: V4 MODEL ON VALIDATION SET")
    log_message("=" * 70)

    val_acc, val_cm, val_report, val_dict, _ = run_evaluation(
        best_v4, val_paths, val_labels, title="V4 Validation"
    )
    log_message(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    log_message(f"Validation Macro F1: {val_dict['macro avg']['f1-score']:.4f}")
    log_message(f"Validation Weighted F1: {val_dict['weighted avg']['f1-score']:.4f}")

    save_confusion_matrix(val_cm, CONFUSION_PLOT, title="Confusion Matrix - Distraction CNN V4")
    log_message(f"Confusion matrix saved: {CONFUSION_PLOT}")

    # 5. Evaluate on Test Set (Friend 5 - Unseen Person)
    log_message("\n" + "=" * 70)
    log_message("EVALUATION: V4 MODEL ON UNSEEN-PERSON TEST SET (Friend 5)")
    log_message("=" * 70)

    test_acc, test_cm, test_report, test_dict, _ = run_evaluation(
        best_v4, test_paths, test_labels, title="V4 Unseen Person Test"
    )
    log_message(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    log_message(f"Test Macro F1: {test_dict['macro avg']['f1-score']:.4f}")
    log_message(f"Test Weighted F1: {test_dict['weighted avg']['f1-score']:.4f}")

    # Save test confusion matrix
    test_confusion_path = CONFUSION_PLOT.replace(".png", "_test.png")
    save_confusion_matrix(test_cm, test_confusion_path, title="Confusion Matrix - V4 Unseen Person Test")
    log_message(f"Test confusion matrix saved: {test_confusion_path}")

    # 6. Write Comprehensive Report
    log_message("\n[6] Writing evaluation reports...")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DRIVER DISTRACTION CNN V4 -- EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Starting Model: {START_MODEL}\n")
        f.write(f"Output Model: {OUTPUT_MODEL}\n")
        f.write(f"Best Model: {BEST_MODEL}\n")
        f.write(f"Confusion Plot: {CONFUSION_PLOT}\n\n")

        f.write("--- 1. DATASET SUMMARY ---\n")
        f.write(f"Train samples: {len(train_paths)}\n")
        f.write(f"Validation samples: {len(val_paths)}\n")
        f.write(f"Test samples (unseen person): {len(test_paths)}\n")
        f.write(f"Train persons: {set(train_persons)}\n")
        f.write(f"Validation persons: {set(val_persons)}\n")
        f.write(f"Test persons: {set(test_persons)}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Seed: {SEED}\n\n")

        f.write("--- 2. TRAINING HISTORY ---\n")
        f.write(f"Phase 1 epochs completed: {len(h1.history['loss'])}\n")
        f.write(f"Phase 1 duration: {p1_duration:.1f} seconds\n")
        f.write(f"Phase 1 best val accuracy: {max(h1.history['val_accuracy']):.4f}\n")
        f.write(f"Phase 2 epochs completed: {len(h2.history['loss'])}\n")
        f.write(f"Phase 2 duration: {p2_duration:.1f} seconds\n")
        f.write(f"Phase 2 best val accuracy: {max(h2.history['val_accuracy']):.4f}\n")
        f.write(f"Total training time: {overall_duration:.1f} seconds\n\n")

        f.write("--- 3. VALIDATION RESULTS ---\n")
        f.write(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
        f.write(f"Validation Macro Precision: {val_dict['macro avg']['precision']:.4f}\n")
        f.write(f"Validation Macro Recall: {val_dict['macro avg']['recall']:.4f}\n")
        f.write(f"Validation Macro F1: {val_dict['macro avg']['f1-score']:.4f}\n")
        f.write(f"Validation Weighted F1: {val_dict['weighted avg']['f1-score']:.4f}\n\n")
        f.write("Classification Report:\n" + val_report + "\n")
        f.write("Confusion Matrix:\n" + str(val_cm) + "\n\n")

        f.write("--- 4. UNSEEN-PERSON TEST RESULTS (Friend 5) ---\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        f.write(f"Test Macro Precision: {test_dict['macro avg']['precision']:.4f}\n")
        f.write(f"Test Macro Recall: {test_dict['macro avg']['recall']:.4f}\n")
        f.write(f"Test Macro F1: {test_dict['macro avg']['f1-score']:.4f}\n")
        f.write(f"Test Weighted F1: {test_dict['weighted avg']['f1-score']:.4f}\n\n")
        f.write("Classification Report:\n" + test_report + "\n")
        f.write("Confusion Matrix:\n" + str(test_cm) + "\n\n")

        f.write("--- 5. PER-CLASS METRICS (Validation) ---\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {name} (c{i}): precision={val_dict[name]['precision']:.4f}, "
                   f"recall={val_dict[name]['recall']:.4f}, f1={val_dict[name]['f1-score']:.4f}\n")
        f.write("\n")

        f.write("--- 6. PER-CLASS METRICS (Test) ---\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {name} (c{i}): precision={test_dict[name]['precision']:.4f}, "
                   f"recall={test_dict[name]['recall']:.4f}, f1={test_dict[name]['f1-score']:.4f}\n")
        f.write("\n")

    # 7. Save machine-readable metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_version": "V4",
        "training": {
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "test_samples": len(test_paths),
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "phase1_epochs": len(h1.history["loss"]),
            "phase1_duration_seconds": p1_duration,
            "phase1_best_val_accuracy": max(h1.history["val_accuracy"]),
            "phase2_epochs": len(h2.history["loss"]),
            "phase2_duration_seconds": p2_duration,
            "phase2_best_val_accuracy": max(h2.history["val_accuracy"]),
            "total_duration_seconds": overall_duration
        },
        "validation": {
            "accuracy": float(val_acc),
            "macro_precision": float(val_dict["macro avg"]["precision"]),
            "macro_recall": float(val_dict["macro avg"]["recall"]),
            "macro_f1": float(val_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(val_dict["weighted avg"]["f1-score"]),
            "per_class": {name: {
                "precision": float(val_dict[name]["precision"]),
                "recall": float(val_dict[name]["recall"]),
                "f1": float(val_dict[name]["f1-score"]),
                "support": int(val_dict[name]["support"])
            } for name in CLASS_NAMES}
        },
        "test_unseen_person": {
            "accuracy": float(test_acc),
            "macro_precision": float(test_dict["macro avg"]["precision"]),
            "macro_recall": float(test_dict["macro avg"]["recall"]),
            "macro_f1": float(test_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(test_dict["weighted avg"]["f1-score"]),
            "per_class": {name: {
                "precision": float(test_dict[name]["precision"]),
                "recall": float(test_dict[name]["recall"]),
                "f1": float(test_dict[name]["f1-score"]),
                "support": int(test_dict[name]["support"])
            } for name in CLASS_NAMES}
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log_message(f"Report written to: {REPORT_PATH}")
    log_message(f"Metrics written to: {METRICS_PATH}")
    log_message(f"History written to: {HISTORY_PATH}")

    log_message("\n" + "=" * 70)
    log_message("V4 TRAINING PIPELINE COMPLETE")
    log_message("=" * 70)

if __name__ == "__main__":
    main()
