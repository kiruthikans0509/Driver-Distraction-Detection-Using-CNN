import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import random

# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = "data/webcam_distraction"
OUTPUT_MODEL = "models/distraction_cnn_webcam_finetuned.keras"
BEST_MODEL = "models/distraction_cnn_webcam_best.keras"

# Same class mapping as test_facemesh.py
CLASS_NAMES = [
    "Safe driving",           # c0
    "Texting - right",        # c1
    "Talking on phone - right",  # c2
    "Texting - left",         # c3
    "Talking on phone - left",   # c4
    "Operating radio",        # c5
    "Drinking",                # c6
    "Reaching behind",        # c7
    "Hair/makeup",             # c8
    "Talking to passenger"    # c9
]

def load_and_split_data():
    """Load webcam dataset and create stratified train/val split"""
    print("Loading webcam dataset...")
    
    # Load all file paths and labels
    file_paths = []
    labels = []
    
    for class_idx in range(10):
        class_dir = os.path.join(DATA_DIR, f"c{class_idx}")
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    file_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_idx)
    
    print(f"Total images loaded: {len(file_paths)}")
    
    # Stratified train/val split (80/20)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, 
        test_size=0.2, 
        random_state=SEED,
        stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Print class distribution
    print("\nClass distribution:")
    for i in range(10):
        train_count = train_labels.count(i)
        val_count = val_labels.count(i)
        total = train_count + val_count
        print(f"  c{i} ({CLASS_NAMES[i]}): {train_count} train, {val_count} val ({total} total)")
    
    return train_paths, val_paths, train_labels, val_labels

def load_and_preprocess_image(path, label):
    """Load and preprocess a single image"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    return img, label

def create_dataset(file_paths, labels, shuffle=True, augment=False):
    """Create a tf.data.Dataset from file paths"""
    # Convert to TensorFlow tensors
    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=SEED)
    
    # Load and preprocess images
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation for training
    if augment:
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.05),  # Small rotation
            layers.RandomTranslation(0.1, 0.1),  # Small translation
            layers.RandomZoom(0.1),  # Small zoom
            # Note: No horizontal flip to preserve left/right actions
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                            num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def load_existing_model():
    """Load the existing distraction model"""
    print("Loading existing model: models/distraction_cnn.keras")
    model = tf.keras.models.load_model("models/distraction_cnn.keras")
    print(f"Model loaded successfully")
    print(f"Total parameters: {model.count_params():,}")
    return model

def freeze_base_layers(model):
    """Freeze MobileNetV2 base layers"""
    for layer in model.layers:
        if layer.name == "mobilenetv2_1.00_224":
            layer.trainable = False
            print(f"Frozen layer: {layer.name}")
        else:
            layer.trainable = True
    
    # Verify trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

def unfreeze_base_layers(model):
    """Unfreeze only the last 25% of MobileNetV2 base layers for fine-tuning"""
    # Get the MobileNetV2 base layer
    mobilenet_layer = None
    for layer in model.layers:
        if layer.name == "mobilenetv2_1.00_224":
            mobilenet_layer = layer
            break
    
    if mobilenet_layer:
        total_layers = len(mobilenet_layer.layers)
        # Unfreeze last 25% of MobileNetV2 layers
        num_trainable = int(total_layers * 0.25)
        num_frozen = total_layers - num_trainable
        
        print(f"MobileNetV2 has {total_layers} sub-layers")
        print(f"Freezing first {num_frozen} layers ({num_frozen/total_layers*100:.1f}%)")
        print(f"Unfreezing last {num_trainable} layers ({num_trainable/total_layers*100:.1f}%)")
        
        # Freeze early layers
        for layer in mobilenet_layer.layers[:num_frozen]:
            layer.trainable = False
        
        # Unfreeze later layers
        for layer in mobilenet_layer.layers[num_frozen:]:
            layer.trainable = True
        
        print(f"Fine-tuning layers: {mobilenet_layer.layers[num_frozen].name} to {mobilenet_layer.layers[-1].name}")
    
    # Ensure custom head layers remain trainable
    for layer in model.layers:
        if layer.name != "mobilenetv2_1.00_224":
            layer.trainable = True
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

def compile_model(model, learning_rate):
    """Compile the model with specified learning rate"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Model compiled with learning rate: {learning_rate}")

def train_model(model, train_ds, val_ds, epochs, phase_name):
    """Train the model with callbacks"""
    print(f"\n=== {phase_name} ===")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            BEST_MODEL,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, val_ds, val_labels):
    """Evaluate model and print detailed metrics"""
    print("\n=== FINAL EVALUATION ===")
    
    # Get predictions
    predictions = model.predict(val_ds, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Convert to numpy arrays
    val_labels_array = np.array(val_labels)
    
    # Overall accuracy
    accuracy = np.mean(predicted_classes == val_labels_array)
    print(f"Overall Validation Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-class Validation Accuracy:")
    for i in range(10):
        class_mask = val_labels_array == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == val_labels_array[class_mask])
            print(f"  c{i} ({CLASS_NAMES[i]}): {class_acc:.4f} ({np.sum(class_mask)} samples)")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels_array, predicted_classes)
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_labels_array, predicted_classes, 
                              target_names=CLASS_NAMES, digits=4))

def main():
    print("=== DISTRACTION MODEL FINE-TUNING ===")
    print("Using webcam dataset for domain adaptation")
    print()
    
    # Load and split data
    train_paths, val_paths, train_labels, val_labels = load_and_split_data()
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds = create_dataset(train_paths, train_labels, shuffle=True, augment=True)
    val_ds = create_dataset(val_paths, val_labels, shuffle=False, augment=False)
    
    # Load existing model
    model = load_existing_model()
    
    # Phase 1: Freeze base layers, train head only
    print("\n=== PHASE 1: TRAINING CLASSIFICATION HEAD ===")
    freeze_base_layers(model)
    compile_model(model, learning_rate=1e-4)
    
    history1 = train_model(model, train_ds, val_ds, epochs=30, phase_name="Phase 1 - Head Training")
    
    # Phase 2: Unfreeze base layers, fine-tune entire model
    print("\n=== PHASE 2: FINE-TUNING ENTIRE MODEL ===")
    unfreeze_base_layers(model)
    compile_model(model, learning_rate=1e-5)  # Lower learning rate for fine-tuning
    
    history2 = train_model(model, train_ds, val_ds, epochs=20, phase_name="Phase 2 - Full Fine-tuning")
    
    # Save final model
    print(f"\nSaving final model to: {OUTPUT_MODEL}")
    model.save(OUTPUT_MODEL)
    print(f"Best model saved to: {BEST_MODEL}")
    
    # Final evaluation
    evaluate_model(model, val_ds, val_labels)
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"Final model: {OUTPUT_MODEL}")
    print(f"Best model: {BEST_MODEL}")

if __name__ == "__main__":
    main()