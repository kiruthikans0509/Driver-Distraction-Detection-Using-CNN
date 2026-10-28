"""
Smoke test for V5 deterministic sampler
Verifies exact budget, class coverage, and sampling distribution
"""
import os
import csv
import random
from collections import defaultdict, Counter

BASE_DIR = r"C:\Users\kiruthikan S\Desktop\driver_monitoring"
TRAIN_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "validation.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "v4_splits", "test_unseen_person.csv")

SEED = 42
EPOCH_BUDGET = 4578
N_COMBINATIONS = 50
BASE_SAMPLES = 91
REMAINDER = EPOCH_BUDGET - (N_COMBINATIONS * BASE_SAMPLES)

random.seed(SEED)

def extract_temporal_group(path):
    filename = os.path.basename(path)
    parts = filename.replace(".jpg", "").split("_")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"

def load_manifest(csv_path):
    paths, labels, persons, temporal_groups = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["image_path"])
            labels.append(int(row["class_id"]))
            persons.append(row["person"])
            temporal_groups.append(extract_temporal_group(row["image_path"]))
    return paths, labels, persons, temporal_groups

def build_person_class_structure(paths, labels, persons, temporal_groups):
    person_class_group_images = defaultdict(lambda: defaultdict(list))
    for path, label, person, group in zip(paths, labels, persons, temporal_groups):
        person_class_group_images[person][label].append({
            "path": path,
            "label": label,
            "group": group
        })
    return person_class_group_images

def calculate_sample_allocation(person_class_group_images):
    allocation = {}
    combinations = []
    
    for person in person_class_group_images:
        for class_id in person_class_group_images[person]:
            combinations.append((person, class_id))
    
    for comb in combinations:
        allocation[comb] = BASE_SAMPLES
    
    for i in range(REMAINDER):
        allocation[combinations[i]] = BASE_SAMPLES + 1
    
    return allocation

def sample_epoch_images(person_class_group_images, allocation):
    epoch_paths = []
    epoch_labels = []
    epoch_persons = []
    epoch_groups = []
    
    for (person, class_id), target_count in allocation.items():
        images = person_class_group_images[person][class_id]
        n_images = len(images)
        
        indices = random.choices(range(n_images), k=target_count)
        for idx in indices:
            img = images[idx]
            epoch_paths.append(img["path"])
            epoch_labels.append(img["label"])
            epoch_persons.append(person)
            epoch_groups.append(img["group"])
    
    combined = list(zip(epoch_paths, epoch_labels, epoch_persons, epoch_groups))
    random.shuffle(combined)
    epoch_paths, epoch_labels, epoch_persons, epoch_groups = zip(*combined)
    
    return list(epoch_paths), list(epoch_labels), epoch_persons, epoch_groups

def check_contamination(train_paths, val_paths, test_paths):
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)
    
    val_in_train = train_set & val_set
    test_in_train = train_set & test_set
    
    return len(val_in_train), len(test_in_train)

print("=" * 70)
print("V5 SAMPLER SMOKE TEST")
print("=" * 70)

print("\nLoading manifests...")
train_paths, train_labels, train_persons, train_groups = load_manifest(TRAIN_CSV)
val_paths, val_labels, val_persons, val_groups = load_manifest(VAL_CSV)
test_paths, test_labels, test_persons, test_groups = load_manifest(TEST_CSV)

print(f"  Training: {len(train_paths)} samples")
print(f"  Validation: {len(val_paths)} samples")
print(f"  Test: {len(test_paths)} samples")

print("\nChecking contamination...")
val_contam, test_contam = check_contamination(train_paths, val_paths, test_paths)
print(f"  Validation images in training: {val_contam}")
print(f"  Test images in training: {test_contam}")

if val_contam > 0 or test_contam > 0:
    print("  ERROR: Contamination detected!")
    exit(1)

print("\nBuilding person-class structure...")
person_class_group_images = build_person_class_structure(train_paths, train_labels, train_persons, train_groups)
print(f"  Found {len(person_class_group_images)} persons")

print("\nCalculating allocation...")
allocation = calculate_sample_allocation(person_class_group_images)
print(f"  Allocation: {sum(1 for v in allocation.values() if v == BASE_SAMPLES)} × {BASE_SAMPLES}, {sum(1 for v in allocation.values() if v == BASE_SAMPLES + 1)} × {BASE_SAMPLES + 1}")
print(f"  Total: {sum(allocation.values())}")

if sum(allocation.values()) != EPOCH_BUDGET:
    print(f"  ERROR: Budget mismatch! Expected {EPOCH_BUDGET}, got {sum(allocation.values())}")
    exit(1)

print("\nSimulating 3 epochs...")
for epoch in range(1, 4):
    print(f"\n  Epoch {epoch}:")
    paths, labels, persons, groups = sample_epoch_images(person_class_group_images, allocation)
    
    print(f"    Total samples: {len(paths)}")
    if len(paths) != EPOCH_BUDGET:
        print(f"    ERROR: Sample count mismatch! Expected {EPOCH_BUDGET}, got {len(paths)}")
        exit(1)
    
    unique_images = len(set(paths))
    repeated = len(paths) - unique_images
    print(f"    Unique images: {unique_images}")
    print(f"    Repeated images: {repeated}")
    
    unique_groups = len(set(groups))
    repeated_groups = len(groups) - unique_groups
    print(f"    Unique groups: {unique_groups}")
    print(f"    Repeated groups: {repeated_groups}")
    
    person_counts = Counter(persons)
    print(f"    Person counts: {dict(person_counts)}")
    
    class_counts = Counter(labels)
    print(f"    Class counts: {dict(class_counts)}")
    
    pc_counts = Counter(zip(persons, labels))
    print(f"    Person-class combinations: {len(pc_counts)}")
    
    if len(class_counts) != 10:
        print(f"    ERROR: Missing classes! Expected 10, got {len(class_counts)}")
        exit(1)
    
    if len(person_counts) != 5:
        print(f"    ERROR: Missing persons! Expected 5, got {len(person_counts)}")
        exit(1)
    
    if len(pc_counts) != 50:
        print(f"    ERROR: Missing person-class combinations! Expected 50, got {len(pc_counts)}")
        exit(1)

print("\n" + "=" * 70)
print("SMOKE TEST PASSED")
print("=" * 70)
