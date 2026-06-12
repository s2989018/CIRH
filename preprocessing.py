import os
import random
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# ---------------- CONFIG ----------------

DATA_DIR = "mirflickr"
IMG_DIR = DATA_DIR
TAG_DIR = os.path.join(DATA_DIR, "meta", "tags")
CLASS_DIR = "mirflickr25k_annotations_v080"

OUT_DIR = "./datasets/MIRFlickr/"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_SIZE = 5000

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MIN_TAG_FREQ = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------
# 1. LOAD TAGS
# -------------------------------

print("Reading tags...")

image_tags = {}

for tag_file in sorted(os.listdir(TAG_DIR)):
    if not tag_file.endswith(".txt"):
        continue

    idx = ''.join(filter(str.isdigit, tag_file))
    path = os.path.join(TAG_DIR, tag_file)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tags = [t.strip() for t in f if t.strip()]

    if tags:
        image_tags[idx] = tags

# -------------------------------
# 2. FILTER TAGS BY FREQUENCY
# -------------------------------

print("Filtering tags by frequency...")

tag_counts = Counter()

for tags in image_tags.values():
    tag_counts.update(set(tags))

frequent_tags = sorted([t for t, c in tag_counts.items() if c >= MIN_TAG_FREQ])
tag_idx = {tag: i for i, tag in enumerate(sorted(frequent_tags))}
DIM_TXT = len(tag_idx)

print("Number of frequent tags:", DIM_TXT)

# -------------------------------
# 3. REMOVE RARE TAGS FROM IMAGES
# -------------------------------

filtered_tags = {}

for idx, tags in image_tags.items():
    kept = [t for t in tags if t in frequent_tags]
    if kept:
        filtered_tags[idx] = kept

# -------------------------------
# 4. LOAD CLASS ANNOTATIONS
# -------------------------------

print("Loading class annotations...")

class_files = sorted([
    f for f in os.listdir(CLASS_DIR)
    if os.path.isfile(os.path.join(CLASS_DIR, f))
    and f.lower().endswith(".txt")
    and not f.lower().endswith("_r1.txt")
    and f.lower() != "readme.txt"
])

DIM_LABEL = len(class_files)

print("DIM_LABEL:", DIM_LABEL)
print("Class files:")
for f in class_files:
    print(" ", f)


class_idx = {c: i for i, c in enumerate(class_files)}

image_labels = {}
valid_ids = set()

for cname in class_files:
    path = os.path.join(CLASS_DIR, cname)

    with open(path, "r") as f:
        for line in f:
            idx = line.strip()

            if idx.isdigit():
                valid_ids.add(idx)

                if idx not in image_labels:
                    image_labels[idx] = np.zeros(DIM_LABEL, dtype=np.float32)

                image_labels[idx][class_idx[cname]] = 1.0
# -------------------------------
# 5. COLLECT VALID SAMPLES LIKE MLHE
# -------------------------------



print("Collecting valid samples...")

samples = []

for idx, tags in image_tags.items():
    if idx not in image_labels:
        continue

    filtered = [t for t in tags if t in frequent_tags]

    if len(filtered) == 0:
        continue

    img_path = os.path.join(IMG_DIR, f"im{idx}.jpg")

    if not os.path.exists(img_path):
        continue

    if image_labels[idx].sum() == 0:
        continue

    samples.append({
        "id": idx,
        "image_path": img_path,
        "tags": filtered,
        "label": image_labels[idx]
    })

samples = sorted(samples, key=lambda x: int(x["id"]))
print("Total valid samples:", len(samples))

id_to_sample = {s["id"]: s for s in samples}
valid_ids = [s["id"] for s in samples]
valid_id_set = set(valid_ids)

print("Num samples:", len(samples))
print("First 20 valid ids:", valid_ids[:20])

# For the rest of CIRH code
final_tags = {s["id"]: s["tags"] for s in samples}
all_ids = valid_ids

print("Images after filtering:", len(all_ids))
print("Number of labels:", DIM_LABEL)

# -------------------------------
# 6. CLASS-BASED SPLIT LIKE MLHE
# -------------------------------

print("Splitting dataset using MLHE-style class-based sampling...")

test_ids = []
train_ids = []
first = True

for label_name in class_files:
    label_pos = class_idx[label_name]

    class_ids = [
        s["id"]
        for s in samples
        if s["label"][label_pos] == 1.0
    ]

    if label_pos == 0:
        print("First class before shuffle:", class_ids[:20])

    random.shuffle(class_ids)

    if label_pos == 0:
        print("First class after shuffle:", class_ids[:20])

    if first:
        test_ids.extend(class_ids[:160])
        train_ids.extend(class_ids[160:160 + 400])
        first = False
    else:
        used = set(test_ids) | set(train_ids)
        class_ids = [idx for idx in class_ids if idx not in used]

        test_ids.extend(class_ids[:80])
        train_ids.extend(class_ids[80:80 + 200])

test_ids = list(dict.fromkeys(test_ids))
train_ids = list(dict.fromkeys(train_ids))

test_set = set(test_ids)

database_ids = [
    idx for idx in valid_ids
    if idx not in test_set
]

train_ids = [
    idx for idx in train_ids
    if idx not in test_set
]

if len(train_ids) < TRAIN_SIZE:
    used = set(train_ids)

    pick = [
        idx for idx in database_ids
        if idx not in used
    ]

    random.shuffle(pick)

    res = TRAIN_SIZE - len(train_ids)
    train_ids.extend(pick[:res])

train_ids = train_ids[:TRAIN_SIZE]

query_ids = test_ids
retrieval_ids = database_ids

print("Query:", len(query_ids))
print("Retrieval:", len(retrieval_ids))
print("Train:", len(train_ids))
print("Query + Retrieval:", len(query_ids) + len(retrieval_ids))
print("Train subset of retrieval:", set(train_ids).issubset(set(retrieval_ids)))
print("Query/retrieval overlap:", len(set(query_ids) & set(retrieval_ids)))
print("Query/train overlap:", len(set(query_ids) & set(train_ids)))

assert set(query_ids).isdisjoint(set(retrieval_ids))
assert set(query_ids).isdisjoint(set(train_ids))
assert set(train_ids).issubset(set(retrieval_ids))
assert len(train_ids) == TRAIN_SIZE

# -------------------------------
# 7. BUILD PAIRS
# -------------------------------

def ids_to_pairs(ids):
    return [
        (os.path.join(IMG_DIR, f"im{idx}.jpg"), final_tags[idx])
        for idx in ids
    ]

query_pairs = ids_to_pairs(query_ids)
retrieval_pairs = ids_to_pairs(retrieval_ids)
train_pairs = ids_to_pairs(train_ids)

# -------------------------------
# 8. BUILD TAG FEATURE
# -------------------------------

def build_tag_vec(tags):
    vec = np.zeros(DIM_TXT, dtype=np.float32)

    for t in tags:
        if t in tag_idx:
            vec[tag_idx[t]] = 1.0

    return vec

# -------------------------------
# 9. LOAD ALEXNET
# -------------------------------

print("Loading AlexNet...")

alexnet = models.alexnet(pretrained=True)
alexnet.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])

alexnet = alexnet.to(DEVICE)
alexnet.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 10. FEATURE EXTRACTION
# -------------------------------

def extract_features(pair_list):
    I = []
    Tvec = []
    L = []
    IDs = []

    batch_imgs = []
    meta = []

    for img_path, tags in tqdm(pair_list):
        idx = ''.join(filter(str.isdigit, os.path.basename(img_path)))

        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        batch_imgs.append(img)
        meta.append((idx, tags))

        if len(batch_imgs) == BATCH_SIZE:
            batch = torch.stack(batch_imgs).to(DEVICE)

            with torch.no_grad():
                feat = alexnet(batch).cpu().numpy()

            for i, (id_, tags_) in enumerate(meta):
                I.append(feat[i])
                Tvec.append(build_tag_vec(tags_))
                L.append(image_labels[id_])
                IDs.append(id_)

            batch_imgs = []
            meta = []

    if batch_imgs:
        batch = torch.stack(batch_imgs).to(DEVICE)

        with torch.no_grad():
            feat = alexnet(batch).cpu().numpy()

        for i, (id_, tags_) in enumerate(meta):
            I.append(feat[i])
            Tvec.append(build_tag_vec(tags_))
            L.append(image_labels[id_])
            IDs.append(id_)

    return (
        np.array(I, dtype=np.float32),
        np.array(Tvec, dtype=np.float32),
        np.array(L, dtype=np.float32),
        np.array(IDs)
    )

# -------------------------------
# 11. EXTRACT DATASETS
# -------------------------------

print("Extracting QUERY features...")
I_te, T_te, L_te, ID_te = extract_features(query_pairs)

print("Extracting DATABASE features...")
I_db, T_db, L_db, ID_db = extract_features(retrieval_pairs)

print("Extracting TRAIN features...")
I_tr, T_tr, L_tr, ID_tr = extract_features(train_pairs)

# -------------------------------
# 12. SAVE DATASETS
# -------------------------------

print("Saving .mat files...")

sio.savemat(os.path.join(OUT_DIR, "mir_query.mat"), {
    "I_te": I_te,
    "T_te": T_te,
    "L_te": L_te,
    "ID_te": ID_te
})

sio.savemat(os.path.join(OUT_DIR, "mir_database.mat"), {
    "I_db": I_db,
    "T_db": T_db,
    "L_db": L_db,
    "ID_db": ID_db
})

sio.savemat(os.path.join(OUT_DIR, "mir_train.mat"), {
    "I_tr": I_tr,
    "T_tr": T_tr,
    "L_tr": L_tr,
    "ID_tr": ID_tr
})

print("Preprocessing finished successfully.")

