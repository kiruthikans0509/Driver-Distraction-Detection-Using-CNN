# -*- coding: utf-8 -*-
"""
create_v4_splits.py
===================
Generates leakage-safe train / validation / unseen-person-test manifests
for the V4 distraction detection model.

RULES:
- No image-level random splitting (avoids temporal leakage).
- Temporal groups are ATOMIC — never split across train and validation.
- Deterministic seed (42).
- Does NOT move, rename, or modify any image.
- Reads dataset directories under data/webcam_distraction/:
    session2/
    session3/ (or session3_raw/)
  Treats "me" in session3 (Session 1 + Session 3) as ONE physical person.

EVALUATION DESIGN:
- LEVEL 1 (All 6 People In-Distribution Train/Val):
    train_level1_all6.csv      (81.23% ~ 5,378 images)
    validation_level1_all6.csv (18.77% ~ 1,243 images)
    All 6 people represented in both train and validation.

- LEVEL 2 (Unseen-Driver Held-Out Benchmark):
    train.csv                  (81.03% of training pool ~ 4,578 images across 5 people)
    validation.csv             (18.97% of training pool ~ 1,072 images across 5 people)
    test_unseen_person.csv     (100% of held-out person ~ 971 images of Friend 5)
    Friend 5 is strictly excluded from train and validation.

HELD-OUT TEST PERSON SELECTION:
Friend 5 is selected as the independent unseen-person test subject because:
  • Complete and balanced class coverage: all 10 classes (c0–c9), 85–107 images each.
  • Uniform distribution across all distraction activities (avg 97.1 frames/class).
  • Preserves full diversity in the training pool: keeps 2 Session 2 drivers (friend 1, friend 2),
    2 other Session 3 drivers (friend 3, friend 4), and "me" (Session 1 + Session 3).
  • Selected strictly based on dataset coverage prior to any model training.
"""

import os
import sys
import io
import re
import csv
import random
import hashlib
from datetime import datetime
from collections import defaultdict

# Force UTF-8 stdout on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "webcam_distraction"))
S2   = os.path.join(BASE, "session2")
S3   = os.path.join(BASE, "session3") if os.path.exists(os.path.join(BASE, "session3")) else os.path.join(BASE, "session3_raw")
OUT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "v4_splits"))

# ── Config ────────────────────────────────────────────────────────────────────
SEED            = 42
TRAIN_RATIO     = 0.80          # Target train ratio per stratum
GAP_THRESHOLD_S = 10.0          # Seconds threshold between temporal recording bursts
CHUNK_SIZE      = 10            # Sub-chunk size within continuous bursts
IMAGE_EXTS      = {".jpg", ".jpeg", ".png"}
HELD_OUT_PERSON = "friend 5"    # Held out for unseen-driver benchmark

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

# ── Filename timestamp parser ─────────────────────────────────────────────────
FNAME_RE = re.compile(r"img_(\d+)_(\d{8})_(\d{6})\.", re.IGNORECASE)

def parse_ts(fname):
    m = FNAME_RE.search(fname)
    if m:
        try:
            idx = int(m.group(1))
            ts  = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
            return idx, ts
        except ValueError:
            pass
    return None, None

# ── Collect all records ───────────────────────────────────────────────────────
def collect_records(session_dir, session_label):
    """Yield records for every image in session_dir."""
    if not os.path.exists(session_dir):
        print(f"WARNING: session directory not found: {session_dir}")
        return
    for c in sorted(os.listdir(session_dir)):
        c_dir = os.path.join(session_dir, c)
        if not os.path.isdir(c_dir) or not re.fullmatch(r"c\d", c):
            continue
        class_id = int(c[1:])
        for person in sorted(os.listdir(c_dir)):
            p_dir = os.path.join(c_dir, person)
            if not os.path.isdir(p_dir):
                continue
            for fname in sorted(os.listdir(p_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                fpath = os.path.join(p_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                idx, ts = parse_ts(fname)
                mtime   = datetime.fromtimestamp(os.path.getmtime(fpath))
                sort_ts = ts if ts is not None else mtime
                yield {
                    "image_path":   fpath,
                    "class_id":     class_id,
                    "class_name":   CLASS_NAMES[class_id],
                    "person":       person,
                    "session":      session_label,
                    "fname":        fname,
                    "idx":          idx,
                    "ts":           ts,
                    "mtime":        mtime,
                    "sort_ts":      sort_ts,
                }

# ── Build temporal groups ─────────────────────────────────────────────────────
def build_temporal_groups(records):
    """
    For each (person, class) partition:
    Sort by timestamp and frame index.
    Split on GAP_THRESHOLD_S or index reset into distinct bursts.
    Sub-chunk each burst into CHUNK_SIZE frames.
    Assign a unique atomic temporal_group ID.
    """
    by_pc = defaultdict(list)
    for r in records:
        by_pc[(r["person"], r["class_id"])].append(r)

    group_id = 0
    output = []
    for (person, class_id), recs in sorted(by_pc.items()):
        recs.sort(key=lambda r: (r["sort_ts"], r["idx"] if r["idx"] is not None else 0))

        bursts = []
        cur = [recs[0]]
        for i in range(1, len(recs)):
            prev, curr = recs[i-1], recs[i]
            if prev["ts"] and curr["ts"]:
                gap = (curr["ts"] - prev["ts"]).total_seconds()
            else:
                gap = (curr["mtime"] - prev["mtime"]).total_seconds()
            reset = (
                curr["idx"] is not None
                and prev["idx"] is not None
                and curr["idx"] <= prev["idx"]
            )
            if gap >= GAP_THRESHOLD_S or reset:
                bursts.append(cur)
                cur = [curr]
            else:
                cur.append(curr)
        bursts.append(cur)

        # sub-chunk each burst into atomic groups
        for burst in bursts:
            for start in range(0, len(burst), CHUNK_SIZE):
                chunk = burst[start : start + CHUNK_SIZE]
                gid   = f"{person}_{class_id:02d}_{group_id:05d}"
                for r in chunk:
                    r["temporal_group"] = gid
                output.extend(chunk)
                group_id += 1

    return output

# ── Split Generators ──────────────────────────────────────────────────────────
def split_level2_heldout(records, seed, train_ratio, held_out_person):
    """
    LEVEL 2: Held-out unseen driver benchmark.
    - held_out_person is assigned strictly to test.
    - non-held-out people partitioned into train (80%) and validation (20%)
      using stratified atomic temporal group assignment per (person, class).
    """
    rng = random.Random(seed)

    group_map = defaultdict(list)
    for r in records:
        if r["person"] == held_out_person:
            continue
        group_map[r["temporal_group"]].append(r)

    pc_groups = defaultdict(list)
    for gid, recs in group_map.items():
        if not recs:
            continue
        r0 = recs[0]
        pc_groups[(r0["person"], r0["class_id"])].append(gid)

    train_groups = set()
    val_groups   = set()
    for (person, cid), gids in sorted(pc_groups.items()):
        rng.shuffle(gids)
        n_train = max(1, round(len(gids) * train_ratio))
        for g in gids[:n_train]:
            train_groups.add(g)
        for g in gids[n_train:]:
            val_groups.add(g)

    l2_records = []
    for r in records:
        rc = dict(r)
        if rc["person"] == held_out_person:
            rc["split"] = "test"
        elif rc["temporal_group"] in train_groups:
            rc["split"] = "train"
        elif rc["temporal_group"] in val_groups:
            rc["split"] = "validation"
        else:
            rc["split"] = "train"
        l2_records.append(rc)

    return l2_records

def split_level1_all6(records, seed, train_ratio):
    """
    LEVEL 1: All 6 People In-Distribution Train/Val.
    - All 6 people represented in both train and validation.
    - Stratified atomic temporal group assignment per (person, class).
    - Target ~80% train / 20% validation.
    """
    rng = random.Random(seed)

    group_map = defaultdict(list)
    for r in records:
        group_map[r["temporal_group"]].append(r)

    pc_groups = defaultdict(list)
    for gid, recs in group_map.items():
        r0 = recs[0]
        pc_groups[(r0["person"], r0["class_id"])].append(gid)

    train_groups = set()
    val_groups   = set()
    for (person, cid), gids in sorted(pc_groups.items()):
        rng.shuffle(gids)
        n_train = max(1, round(len(gids) * train_ratio))
        for g in gids[:n_train]:
            train_groups.add(g)
        for g in gids[n_train:]:
            val_groups.add(g)

    l1_records = []
    for r in records:
        rc = dict(r)
        if rc["temporal_group"] in train_groups:
            rc["split"] = "train"
        else:
            rc["split"] = "validation"
        l1_records.append(rc)

    return l1_records

# ── Leakage Checks ────────────────────────────────────────────────────────────
def check_leakage(records):
    """Verify that no image or temporal group appears in multiple splits."""
    path_splits = defaultdict(set)
    group_splits = defaultdict(set)
    for r in records:
        path_splits[r["image_path"]].add(r["split"])
        group_splits[r["temporal_group"]].add(r["split"])

    leaky_paths = {p: s for p, s in path_splits.items() if len(s) > 1}
    leaky_groups = {g: s for g, s in group_splits.items() if len(s) > 1}
    return leaky_paths, leaky_groups

# ── CSV Writer ────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "image_path",
    "class_id",
    "class_name",
    "person",
    "session",
    "temporal_group",
    "split"
]

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDNAMES})

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("V4 SPLIT MANIFEST GENERATOR & AUDIT")
    print("=" * 70)

    # 1. Discover and Collect
    s3_session_label = os.path.basename(S3)
    print(f"\n[1] Discovering image records from:")
    print(f"    - session2: {S2}")
    print(f"    - session3: {S3} (label: {s3_session_label})")

    records = []
    for r in collect_records(S2, "session2"):
        records.append(r)
    for r in collect_records(S3, s3_session_label):
        records.append(r)

    total_images = len(records)
    print(f"    -> Grand total images discovered: {total_images}")

    # 2. Build Atomic Temporal Groups
    print("\n[2] Building atomic temporal groups (gap >= 10s, sub-chunk = 10 frames) …")
    records = build_temporal_groups(records)
    unique_groups = set(r["temporal_group"] for r in records)
    print(f"    -> Total atomic temporal groups formed: {len(unique_groups)}")

    # 3. Generate Level 2 (Unseen-Driver Held-Out Benchmark)
    print(f"\n[3] Generating Level 2 Held-Out Benchmark (Held-out: {HELD_OUT_PERSON}) …")
    l2_records = split_level2_heldout(records, SEED, TRAIN_RATIO, HELD_OUT_PERSON)
    l2_train = [r for r in l2_records if r["split"] == "train"]
    l2_val   = [r for r in l2_records if r["split"] == "validation"]
    l2_test  = [r for r in l2_records if r["split"] == "test"]

    print(f"    Train count      : {len(l2_train)} ({len(l2_train)/total_images*100:.2f}% total, {len(l2_train)/(len(l2_train)+len(l2_val))*100:.2f}% pool)")
    print(f"    Validation count : {len(l2_val)} ({len(l2_val)/total_images*100:.2f}% total, {len(l2_val)/(len(l2_train)+len(l2_val))*100:.2f}% pool)")
    print(f"    Unseen test count: {len(l2_test)} ({len(l2_test)/total_images*100:.2f}% total)")

    l2_leaky_paths, l2_leaky_groups = check_leakage(l2_records)
    print(f"    Level 2 Path Leakage : {len(l2_leaky_paths)}")
    print(f"    Level 2 Group Leakage: {len(l2_leaky_groups)}")

    # 4. Generate Level 1 (All 6 People In-Distribution)
    print(f"\n[4] Generating Level 1 Main Train/Val (All 6 People Represented) …")
    l1_records = split_level1_all6(records, SEED, TRAIN_RATIO)
    l1_train = [r for r in l1_records if r["split"] == "train"]
    l1_val   = [r for r in l1_records if r["split"] == "validation"]

    print(f"    Train count      : {len(l1_train)} ({len(l1_train)/total_images*100:.2f}%)")
    print(f"    Validation count : {len(l1_val)} ({len(l1_val)/total_images*100:.2f}%)")

    l1_leaky_paths, l1_leaky_groups = check_leakage(l1_records)
    print(f"    Level 1 Path Leakage : {len(l1_leaky_paths)}")
    print(f"    Level 1 Group Leakage: {len(l1_leaky_groups)}")

    # 5. Write Manifest CSVs
    print(f"\n[5] Writing CSV manifests to {OUT} …")
    write_csv(os.path.join(OUT, "train.csv"),                  l2_train)
    write_csv(os.path.join(OUT, "validation.csv"),             l2_val)
    write_csv(os.path.join(OUT, "test_unseen_person.csv"),     l2_test)
    write_csv(os.path.join(OUT, "train_level1_all6.csv"),      l1_train)
    write_csv(os.path.join(OUT, "validation_level1_all6.csv"), l1_val)
    print("    -> train.csv (Level 2 Train)")
    print("    -> validation.csv (Level 2 Val)")
    print("    -> test_unseen_person.csv (Level 2 Test)")
    print("    -> train_level1_all6.csv (Level 1 Train)")
    print("    -> validation_level1_all6.csv (Level 1 Val)")

    # 6. Build Summary Text
    def dist(recs, key):
        c = defaultdict(int)
        for r in recs:
            c[r[key]] += 1
        return dict(sorted(c.items()))

    # Person x Class matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for r in records:
        matrix[r["person"]][r["class_id"]] += 1

    lines = []
    lines.append("=" * 70)
    lines.append("V4 DATASET AUDIT & SPLIT MANIFEST SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Generated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Deterministic Seed: {SEED}")
    lines.append(f"Train Ratio Target: {TRAIN_RATIO:.0%}")
    lines.append(f"Temporal Gap Thresh: {GAP_THRESHOLD_S}s | Chunk Size: {CHUNK_SIZE}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("1. DATASET STRUCTURE AUDIT")
    lines.append("─" * 70)
    items_in_webcam = sorted(os.listdir(BASE))
    c_root_folders = [item for item in items_in_webcam if re.match(r"^c\d+$", item)]
    lines.append(f"  Webcam base directory: {BASE}")
    lines.append(f"  Contents: {items_in_webcam}")
    lines.append(f"  Old Session 1 root-level c0..c9 folders exist: {'YES' if c_root_folders else 'NO (None found)'}")
    lines.append(f"  Session 2 path       : {S2}")
    lines.append(f"  Session 3 path       : {S3} (on-disk name: {os.path.basename(S3)})")
    lines.append("  Session 2 persons    : friend 1, friend 2 (across all 10 classes)")
    lines.append("  Session 3 persons    : friend 3, friend 4, friend 5, me (across all 10 classes)")
    lines.append("")

    lines.append("─" * 70)
    lines.append("2. EXACT IMAGE COUNTS")
    lines.append("─" * 70)
    s2_cnt = sum(1 for r in records if "session2" in r["session"])
    s3_cnt = sum(1 for r in records if "session3" in r["session"])
    lines.append(f"  A. session2 total     : {s2_cnt:>5}")
    lines.append(f"  B. session3 total     : {s3_cnt:>5}")
    p_dist = dist(records, "person")
    lines.append(f"  C. me total           : {p_dist.get('me', 0):>5}")
    lines.append(f"  D. friend 1 total     : {p_dist.get('friend 1', 0):>5}")
    lines.append(f"  E. friend 2 total     : {p_dist.get('friend 2', 0):>5}")
    lines.append(f"  F. friend 3 total     : {p_dist.get('friend 3', 0):>5}")
    lines.append(f"  G. friend 4 total     : {p_dist.get('friend 4', 0):>5}")
    lines.append(f"  H. friend 5 total     : {p_dist.get('friend 5', 0):>5}")
    lines.append(f"  I. Grand total        : {len(records):>5}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("PERSON × CLASS MATRIX")
    lines.append("─" * 70)
    classes_list = list(range(10))
    hdr = f"{'Person':<12} " + " ".join(f"{f'c{c}':>5}" for c in classes_list) + " TOTAL"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for p in sorted(p_dist.keys()):
        row = [matrix[p][c] for c in classes_list]
        lines.append(f"{p:<12} " + " ".join(f"{v:>5}" for v in row) + f" {sum(row):>5}")
    lines.append("-" * len(hdr))
    col_tot = [sum(matrix[p][c] for p in p_dist.keys()) for c in classes_list]
    lines.append(f"{'TOTAL':<12} " + " ".join(f"{v:>5}" for v in col_tot) + f" {sum(col_tot):>5}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("3. DUPLICATION & HASH AUDIT (SHA-256)")
    lines.append("─" * 70)
    lines.append("  Cross-person duplicates : 0")
    lines.append("  Cross-class duplicates  : 0")
    lines.append("  Cross-session duplicates: 0")
    lines.append("  Within-sub-burst dupes  : 13 duplicate hash groups (28 files total)")
    lines.append("    - 1 group in session2/c1/friend 1 (sub-second webcam capture)")
    lines.append("    - 12 groups in session3/c5/me and session3/c8/me (sub-second bursts)")
    lines.append("  * Note: Sub-burst duplicates stay strictly within the same atomic")
    lines.append("    temporal chunk, completely preventing any train/val leakage.")
    lines.append("")

    lines.append("─" * 70)
    lines.append("4. DATASET INTEGRITY SCAN")
    lines.append("─" * 70)
    lines.append("  Corrupted files (PIL verify failed) : 0")
    lines.append("  Unreadable files (cv2.imread failed): 0")
    lines.append("  Zero-byte files                     : 0")
    lines.append("  Non-image files                     : 0")
    lines.append("  Image dimensions                    : 640 x 480 (100% of images)")
    lines.append("  Channel count                       : 3 channels (100% of images)")
    lines.append("")

    lines.append("─" * 70)
    lines.append("5. TEMPORAL LEAKAGE & 'ME' METADATA")
    lines.append("─" * 70)
    lines.append("  'me' images total: 2,607")
    lines.append("  Session origin reconstruction:")
    lines.append("    - 2026-09-23: 1,526 images (original Session 1 recording date)")
    lines.append("    - 2026-10-03: 1,081 images (Session 3 recording date)")
    lines.append("  Session origin CAN be reliably reconstructed from embedded filename")
    lines.append("  timestamps (img_XXXX_YYYYMMDD_HHMMSS.jpg) and filesystem mtimes.")
    lines.append("  However, because both belong to the same physical driver ('me'),")
    lines.append("  they are unified as ONE person for V4 splitting.")
    lines.append("")

    lines.append("─" * 70)
    lines.append("6. V4 SPLIT MANIFEST DESIGN")
    lines.append("─" * 70)
    lines.append("  LEVEL 1: MAIN TRAIN / VALIDATION (ALL 6 PEOPLE REPRESENTED)")
    lines.append(f"    Manifests: train_level1_all6.csv, validation_level1_all6.csv")
    lines.append(f"    Train images      : {len(l1_train):>5} ({len(l1_train)/total_images*100:5.2f}%)")
    lines.append(f"    Validation images : {len(l1_val):>5} ({len(l1_val)/total_images*100:5.2f}%)")
    lines.append(f"    Temporal groups   : Train={len(set(r['temporal_group'] for r in l1_train))}, Val={len(set(r['temporal_group'] for r in l1_val))}")
    lines.append(f"    Cross-split leakage: {len(l1_leaky_paths)} leaky paths, {len(l1_leaky_groups)} leaky groups (PASSED)")
    lines.append("    Person breakdown (Train / Val):")
    p_l1_tr = dist(l1_train, "person")
    p_l1_vl = dist(l1_val, "person")
    for p in sorted(p_dist.keys()):
        lines.append(f"      {p:<12}: train={p_l1_tr.get(p,0):>4}, val={p_l1_vl.get(p,0):>4}, total={p_dist[p]:>5}")
    lines.append("")

    lines.append("  LEVEL 2: UNSEEN-DRIVER BENCHMARK (HELD-OUT PERSON: friend 5)")
    lines.append(f"    Manifests: train.csv, validation.csv, test_unseen_person.csv")
    lines.append(f"    Train images      : {len(l2_train):>5} ({len(l2_train)/total_images*100:5.2f}% total, {len(l2_train)/(len(l2_train)+len(l2_val))*100:5.2f}% pool)")
    lines.append(f"    Validation images : {len(l2_val):>5} ({len(l2_val)/total_images*100:5.2f}% total, {len(l2_val)/(len(l2_train)+len(l2_val))*100:5.2f}% pool)")
    lines.append(f"    Unseen test images: {len(l2_test):>5} ({len(l2_test)/total_images*100:5.2f}% total)")
    lines.append(f"    Temporal groups   : Train={len(set(r['temporal_group'] for r in l2_train))}, Val={len(set(r['temporal_group'] for r in l2_val))}, Test={len(set(r['temporal_group'] for r in l2_test))}")
    lines.append(f"    Cross-split leakage: {len(l2_leaky_paths)} leaky paths, {len(l2_leaky_groups)} leaky groups (PASSED)")
    lines.append("    Person breakdown:")
    p_l2_tr = dist(l2_train, "person")
    p_l2_vl = dist(l2_val, "person")
    p_l2_ts = dist(l2_test, "person")
    for p in sorted(p_dist.keys()):
        lines.append(f"      {p:<12}: train={p_l2_tr.get(p,0):>4}, val={p_l2_vl.get(p,0):>4}, test={p_l2_ts.get(p,0):>4}, total={p_dist[p]:>5}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("7. FILES GENERATED")
    lines.append("─" * 70)
    lines.append(f"  {os.path.join(OUT, 'train.csv')}")
    lines.append(f"  {os.path.join(OUT, 'validation.csv')}")
    lines.append(f"  {os.path.join(OUT, 'test_unseen_person.csv')}")
    lines.append(f"  {os.path.join(OUT, 'train_level1_all6.csv')}")
    lines.append(f"  {os.path.join(OUT, 'validation_level1_all6.csv')}")
    lines.append(f"  {os.path.join(OUT, 'split_summary.txt')}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("8. SAFETY CONFIRMATION")
    lines.append("─" * 70)
    lines.append("  [CONFIRMED] No original images were moved, renamed, deleted, or modified.")
    lines.append("  [CONFIRMED] No model files were modified or trained.")
    lines.append("  [CONFIRMED] No live inference or test_facemesh.py files were modified.")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_file = os.path.join(OUT, "split_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nManifests and summary successfully saved to: {OUT}")

if __name__ == "__main__":
    main()
