"""
V5 Training Script - Experiment D (Combined Approach)
Deterministic person-group-class balanced sampling with enhanced augmentation
"""
import os
import sys
import io
import csv
import json
import time
import random
from collections import defaultdict
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 70)
print("V5 TRAINING - EXPERIMENT D (COMBINED APPROACH)")
print("=" * 70)

BASE_DIR = r"C:\Users\kiruthikan S\Desktop\driver_monitoring"
V4_MODEL = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v4_best.keras")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "validation.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "test_unseen_person.csv")

V5_MODEL = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v5.keras")
V5_BEST = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v5_best.keras")
V5_P1_BEST = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v5_p1_best.keras")
V5_P2_BEST = os.path.join(BASE_DIR, "models", "distraction_cnn_webcam_v5_p2_best.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCH_BUDGET = 4578
N_COMBINATIONS = 50
BASE_SAMPLES = 91
REMAINDER = EPOCH_BUDGET - (N_COMBINATIONS * BASE_SAMPLES)  # 28

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [
    "Safe driving",
    "Texting - right",
    "Talking on phone - right",
    "Texting - left",
    "Talking on phone - left",
    "Operating radio",
    "Drinking",
    "Reaching behind",
    "Hair/makeup",
    "Talking to passenger",
]

def log_message(msg):
    """Log message to both stdout and file."""
    print(msg)
    with open(os.path.join(BASE_DIR, "models", "distraction_v5_training.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def extract_temporal_group(path):
    """Extract temporal group identifier from filename."""
    filename = os.path.basename(path)
    parts = filename.replace(".jpg", "").split("_")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"

def load_manifest(csv_path):
    """Load manifest and return paths, labels, persons, temporal_groups."""
    paths, labels, persons, temporal_groups = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["image_path"])
            labels.append(int(row["class_id"]))
            persons.append(row["person"])
            temporal_groups.append(extract_temporal_group(row["image_path"]))
    return paths, labels, persons, temporal_groups

def load_and_preprocess(path, label):
    """Load and preprocess a single image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    return img, label

def make_augmentation_layer():
    """Create data augmentation layer with enhanced brightness/contrast."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.30),  # Enhanced from 0.15
        layers.RandomContrast(0.30),  # Enhanced from 0.15
    ], name="augmentation")

def build_person_class_structure(paths, labels, persons, temporal_groups):
    """Build structure for deterministic sampling."""
    person_class_group_images = defaultdict(lambda: defaultdict(list))
    for path, label, person, group in zip(paths, labels, persons, temporal_groups):
        person_class_group_images[person][label].append({
            "path": path,
            "label": label,
            "group": group
        })
    return person_class_group_images

def calculate_sample_allocation(person_class_group_images):
    """Calculate deterministic sample allocation: 91 or 92 per person-class combination."""
    allocation = {}
    combinations = []
    
    # Collect all person-class combinations
    for person in person_class_group_images:
        for class_id in person_class_group_images[person]:
            combinations.append((person, class_id))
    
    # Base allocation: 91 for all
    for comb in combinations:
        allocation[comb] = BASE_SAMPLES
    
    # Distribute remainder: 28 combinations get +1
    for i in range(REMAINDER):
        allocation[combinations[i]] = BASE_SAMPLES + 1
    
    return allocation

def sample_epoch_images(person_class_group_images, allocation):
    """Generate deterministic sample list for one epoch."""
    epoch_paths = []
    epoch_labels = []
    epoch_persons = []
    epoch_groups = []
    
    for (person, class_id), target_count in allocation.items():
        images = person_class_group_images[person][class_id]
        n_images = len(images)
        
        # Sample with replacement if needed
        indices = random.choices(range(n_images), k=target_count)
        for idx in indices:
            img = images[idx]
            epoch_paths.append(img["path"])
            epoch_labels.append(img["label"])
            epoch_persons.append(person)
            epoch_groups.append(img["group"])
    
    # Shuffle the combined list
    combined = list(zip(epoch_paths, epoch_labels, epoch_persons, epoch_groups))
    random.shuffle(combined)
    epoch_paths, epoch_labels, epoch_persons, epoch_groups = zip(*combined)
    
    return list(epoch_paths), list(epoch_labels), epoch_persons, epoch_groups

def log_sampling_stats(paths, labels, persons, groups):
    """Log sampling statistics for an epoch."""
    from collections import Counter
    log_message(f"  Total samples: {len(paths)}")
    log_message(f"  Unique images: {len(set(paths))}")
    log_message(f"  Repeated images: {len(paths) - len(set(paths))}")
    log_message(f"  Unique groups: {len(set(groups))}")
    log_message(f"  Repeated groups: {len(groups) - len(set(groups))}")
    
    person_counts = Counter(persons)
    log_message(f"  Person counts: {dict(person_counts)}")
    
    class_counts = Counter(labels)
    log_message(f"  Class counts: {dict(class_counts)}")
    
    pc_counts = Counter(zip(persons, labels))
    log_message(f"  Person-class combinations: {len(pc_counts)}")

def create_balanced_dataset(person_class_group_images, allocation, augment=False):
    """Create dataset with deterministic sampling per epoch."""
    # Generate initial epoch sample list
    paths, labels, persons, groups = sample_epoch_images(person_class_group_images, allocation)
    
    # Log initial sampling stats
    log_message("  Initial epoch sampling statistics:")
    log_sampling_stats(paths, labels, persons, groups)
    
    # Create dataset from sample list
    paths_t = tf.constant(paths)
    labels_t = tf.constant(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        aug = make_augmentation_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return ds, len(paths), person_class_group_images, allocation

def epoch_dataset_generator(person_class_group_images, allocation, augment=True):
    """Generator that yields batches with fresh sampling each epoch."""
    epoch_num = 0
    while True:
        epoch_num += 1
        if epoch_num % 5 == 1:  # Log every 5 epochs
            log_message(f"  Regenerating epoch {epoch_num} sample list...")
        
        paths, labels, persons, groups = sample_epoch_images(person_class_group_images, allocation)
        
        if epoch_num % 5 == 1:
            log_message(f"  Epoch {epoch_num} sampling statistics:")
            log_sampling_stats(paths, labels, persons, groups)
        
        paths_t = tf.constant(paths)
        labels_t = tf.constant(labels)
        ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if augment:
            aug = make_augmentation_layer()
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Yield batches from the dataset
        for batch in ds:
            yield batch

def save_augmentation_samples(dataset, save_path):
    """Save sample augmented images for inspection."""
    # Get one batch
    for batch in dataset.take(1):
        images, labels = batch
        break
    
    # Take first 8 images
    sample_images = images[:8]
    sample_labels = labels[:8]
    
    # Denormalize for visualization (reverse preprocess_input approximately)
    sample_images = (sample_images + 1.0) * 127.5
    sample_images = tf.clip_by_value(sample_images, 0, 255)
    sample_images = tf.cast(sample_images, tf.uint8)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
        row, col = i // 4, i % 4
        axes[row, col].imshow(img.numpy())
        axes[row, col].set_title(f"Class {label.numpy()} ({CLASS_NAMES[label.numpy()]})")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

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

class MacroF1Callback(tf.keras.callbacks.Callback):
    """Custom callback to save best model based on validation macro F1."""
    def __init__(self, validation_data, filepath):
        super().__init__()
        self.validation_data = validation_data
        self.filepath = filepath
        self.best_macro_f1 = -1.0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        val_ds = self.validation_data
        y_true = []
        y_pred = []
        
        for x_batch, y_batch in val_ds:
            y_true.extend(y_batch.numpy())
            pred = self.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(pred, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate macro F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        logs = logs or {}
        logs['val_macro_f1'] = macro_f1
        
        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.filepath)
            print(f"\n  New best validation macro F1: {macro_f1:.4f} - saving checkpoint")

def train_two_phases(model, val_ds, person_class_group_images, allocation):
    """Train model in two phases with epoch-wise sampling refresh."""
    # Phase 1
    log_message("=" * 70)
    log_message("PHASE 1: Training classification head (MobileNetV2 frozen)")
    log_message("=" * 70)
    freeze_phase1(model)
    compile_model(model, lr=5e-5)
    
    p1_weights_path = V5_P1_BEST.replace(".keras", ".weights.h5")
    cb1 = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        MacroF1Callback(val_ds, p1_weights_path)
    ]
    
    # Create generator for training data
    train_gen = epoch_dataset_generator(person_class_group_images, allocation, augment=True)
    
    p1_start = time.time()
    h1 = model.fit(train_gen, validation_data=val_ds, epochs=15, callbacks=cb1, verbose=1, steps_per_epoch=286, shuffle=False)
    p1_duration = time.time() - p1_start
    log_message(f"Phase 1 completed in {p1_duration:.1f} seconds ({p1_duration/60:.1f} minutes)")
    
    # Save Phase 1 best model
    p1_best_macro_f1 = max(h1.history.get("val_macro_f1", [-1]))
    log_message(f"Phase 1 best validation macro F1: {p1_best_macro_f1:.4f}")
    model.load_weights(p1_weights_path)
    model.save(V5_P1_BEST)
    
    # Phase 2
    log_message("=" * 70)
    log_message("PHASE 2: Fine-tuning last 25% of MobileNetV2")
    log_message("=" * 70)
    unfreeze_phase2(model)
    compile_model(model, lr=5e-6)
    
    p2_weights_path = V5_P2_BEST.replace(".keras", ".weights.h5")
    cb2 = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        MacroF1Callback(val_ds, p2_weights_path)
    ]
    
    p2_start = time.time()
    h2 = model.fit(train_gen, validation_data=val_ds, epochs=15, callbacks=cb2, verbose=1, steps_per_epoch=286, shuffle=False)
    p2_duration = time.time() - p2_start
    log_message(f"Phase 2 completed in {p2_duration:.1f} seconds ({p2_duration/60:.1f} minutes)")
    
    # Save Phase 2 best model
    p2_best_macro_f1 = max(h2.history.get("val_macro_f1", [-1]))
    log_message(f"Phase 2 best validation macro F1: {p2_best_macro_f1:.4f}")
    model.load_weights(p2_weights_path)
    model.save(V5_P2_BEST)
    
    # Select best checkpoint by macro F1
    if p2_best_macro_f1 >= p1_best_macro_f1:
        log_message(f"Selecting Phase 2 checkpoint as overall best ({p2_best_macro_f1:.4f} >= {p1_best_macro_f1:.4f})")
        best_source = V5_P2_BEST
    else:
        log_message(f"Selecting Phase 1 checkpoint as overall best ({p1_best_macro_f1:.4f} > {p2_best_macro_f1:.4f})")
        best_source = V5_P1_BEST
    
    # Copy to final best path
    import shutil
    shutil.copy(best_source, V5_BEST)
    log_message(f"Best checkpoint saved to: {V5_BEST}")
    
    # Clean up temporary files
    if os.path.exists(p1_weights_path):
        os.remove(p1_weights_path)
    if os.path.exists(p2_weights_path):
        os.remove(p2_weights_path)
    if os.path.exists(V5_P1_BEST):
        os.remove(V5_P1_BEST)
    if os.path.exists(V5_P2_BEST):
        os.remove(V5_P2_BEST)
    
    return h1, h2
    # Phase 1
    log_message("=" * 70)
    log_message("PHASE 1: Training classification head (MobileNetV2 frozen)")
    log_message("=" * 70)
    freeze_phase1(model)
    compile_model(model, lr=5e-5)
    
    cb1 = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(V5_P1_BEST, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    ]
    
    p1_start = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cb1, verbose=1)
    p1_duration = time.time() - p1_start
    log_message(f"Phase 1 completed in {p1_duration:.1f} seconds ({p1_duration/60:.1f} minutes)")
    
    # Phase 2
    log_message("=" * 70)
    log_message("PHASE 2: Fine-tuning last 25% of MobileNetV2")
    log_message("=" * 70)
    unfreeze_phase2(model)
    compile_model(model, lr=5e-6)
    
    cb2 = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(V5_P2_BEST, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    ]
    
    p2_start = time.time()
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cb2, verbose=1)
    p2_duration = time.time() - p2_start
    log_message(f"Phase 2 completed in {p2_duration:.1f} seconds ({p2_duration/60:.1f} minutes)")
    
    # Select best checkpoint
    max_p1 = max(h1.history.get("val_accuracy", [-1]))
    max_p2 = max(h2.history.get("val_accuracy", [-1]))
    
    if max_p2 >= max_p1 and os.path.exists(V5_P2_BEST):
        log_message(f"Selecting Phase 2 checkpoint as overall best ({max_p2:.4f} >= {max_p1:.4f})")
        best_source = V5_P2_BEST
    else:
        log_message(f"Selecting Phase 1 checkpoint as overall best ({max_p1:.4f} > {max_p2:.4f})")
        best_source = V5_P1_BEST
    
    # Copy to final best path
    import shutil
    shutil.copy(best_source, V5_BEST)
    log_message(f"Best checkpoint saved to: {V5_BEST}")
    
    # Clean up temporary checkpoints
    if os.path.exists(V5_P1_BEST):
        os.remove(V5_P1_BEST)
    if os.path.exists(V5_P2_BEST):
        os.remove(V5_P2_BEST)
    
    return h1, h2

def save_history_csv(h1, h2):
    """Save training history to CSV."""
    history = []
    for i, (loss, acc, val_loss, val_acc) in enumerate(zip(
        h1.history["loss"], h1.history["accuracy"],
        h1.history["val_loss"], h1.history["val_accuracy"]
    )):
        history.append({"epoch": i+1, "loss": loss, "accuracy": acc, "val_loss": val_loss, "val_accuracy": val_acc, "phase": "phase1"})
    
    for i, (loss, acc, val_loss, val_acc) in enumerate(zip(
        h2.history["loss"], h2.history["accuracy"],
        h2.history["val_loss"], h2.history["val_accuracy"]
    )):
        history.append({"epoch": i+1, "loss": loss, "accuracy": acc, "val_loss": val_loss, "val_accuracy": val_acc, "phase": "phase2"})
    
    with open(os.path.join(BASE_DIR, "models", "distraction_v5_history.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "phase"])
        writer.writeheader()
        writer.writerows(history)

def plot_training_curves(h1, h2):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(h1.history["loss"], label="Phase 1 Train")
    axes[0].plot(h1.history["val_loss"], label="Phase 1 Val")
    axes[0].plot(h2.history["loss"], label="Phase 2 Train")
    axes[0].plot(h2.history["val_loss"], label="Phase 2 Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(h1.history["accuracy"], label="Phase 1 Train")
    axes[1].plot(h1.history["val_accuracy"], label="Phase 1 Val")
    axes[1].plot(h2.history["accuracy"], label="Phase 2 Train")
    axes[1].plot(h2.history["val_accuracy"], label="Phase 2 Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "models", "distraction_v5_training_curves.png"), dpi=150)
    plt.close()

# Main training loop
log_message("Loading training data...")
train_paths, train_labels, train_persons, train_groups = load_manifest(TRAIN_CSV)
log_message(f"  Loaded {len(train_paths)} training samples")

log_message("Loading validation data...")
val_paths, val_labels, val_persons, val_groups = load_manifest(VAL_CSV)
log_message(f"  Loaded {len(val_paths)} validation samples")

log_message("Building person-class-group structure...")
person_class_group_images = build_person_class_structure(train_paths, train_labels, train_persons, train_groups)
log_message(f"  Found {len(person_class_group_images)} persons")

log_message("Calculating deterministic sample allocation...")
allocation = calculate_sample_allocation(person_class_group_images)
log_message(f"  Allocation: {sum(1 for v in allocation.values() if v == BASE_SAMPLES)} combinations × {BASE_SAMPLES}, {sum(1 for v in allocation.values() if v == BASE_SAMPLES + 1)} combinations × {BASE_SAMPLES + 1}")
log_message(f"  Total: {sum(allocation.values())} samples")

log_message("Creating sample dataset for augmentation visualization...")
train_ds_vis, actual_samples, _, _ = create_balanced_dataset(person_class_group_images, allocation, augment=True)
log_message(f"  Created dataset with {actual_samples} samples per epoch")
log_message(f"  Expected: {EPOCH_BUDGET} samples")

log_message("Creating validation dataset...")
val_paths_t = tf.constant(val_paths)
val_labels_t = tf.constant(val_labels)
val_ds = tf.data.Dataset.from_tensor_slices((val_paths_t, val_labels_t))
val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
log_message(f"  Created validation dataset")

log_message("Saving augmentation samples...")
save_augmentation_samples(train_ds_vis, os.path.join(BASE_DIR, "models", "distraction_v5_augmentation_samples.png"))
log_message(f"  Saved to: models/distraction_v5_augmentation_samples.png")

log_message("Loading V4 checkpoint...")
model = tf.keras.models.load_model(V4_MODEL)
log_message(f"  Loaded from: {V4_MODEL}")

log_message("Starting training...")
h1, h2 = train_two_phases(model, val_ds, person_class_group_images, allocation)

log_message("Saving training history...")
save_history_csv(h1, h2)
log_message(f"  Saved to: models/distraction_v5_history.csv")

log_message("Plotting training curves...")
plot_training_curves(h1, h2)
log_message(f"  Saved to: models/distraction_v5_training_curves.png")

log_message("Saving final model...")
model.save(V5_MODEL)
log_message(f"  Saved to: {V5_MODEL}")

log_message("=" * 70)
log_message("TRAINING COMPLETE")
log_message("=" * 70)
