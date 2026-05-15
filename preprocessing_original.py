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

QUERY_SIZE = 2000
RETRIEVAL_SIZE = 18015
TRAIN_SIZE = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MIN_TAG_FREQ = 20

# -------------------------------
# 1️⃣ LOAD TAGS
# -------------------------------

print("Reading tags...")

image_tags = {}

for tag_file in os.listdir(TAG_DIR):

    if not tag_file.endswith(".txt"):
        continue

    idx = ''.join(filter(str.isdigit, tag_file))

    path = os.path.join(TAG_DIR, tag_file)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:

        tags = [t.strip() for t in f if t.strip()]

        if tags:
            image_tags[idx] = tags


# -------------------------------
# 2️⃣ FILTER TAGS BY FREQUENCY
# -------------------------------

print("Filtering tags by frequency...")

tag_counts = Counter()

for tags in image_tags.values():
    tag_counts.update(set(tags))

frequent_tags = {t for t, c in tag_counts.items() if c >= MIN_TAG_FREQ}

tag_idx = {tag: i for i, tag in enumerate(sorted(frequent_tags))}

DIM_TXT = len(tag_idx)

print("Number of frequent tags:", DIM_TXT)


# -------------------------------
# 3️⃣ REMOVE RARE TAGS FROM IMAGES
# -------------------------------

filtered_tags = {}

for idx, tags in image_tags.items():

    kept = [t for t in tags if t in frequent_tags]

    if kept:
        filtered_tags[idx] = kept


# -------------------------------
# 4️⃣ LOAD CLASS ANNOTATIONS
# -------------------------------

print("Loading class annotations...")

class_files = sorted(os.listdir(CLASS_DIR))

class_idx = {c: i for i, c in enumerate(class_files)}

DIM_LABEL = len(class_idx)

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
                    image_labels[idx] = np.zeros(DIM_LABEL)

                image_labels[idx][class_idx[cname]] = 1


# -------------------------------
# 5️⃣ REMOVE IMAGES WITHOUT LABELS
# -------------------------------

final_tags = {idx: tags for idx, tags in filtered_tags.items() if idx in valid_ids}

print("Images after filtering:", len(final_tags))

# -------------------------------
# 5.5 COMPUTE IDF WEIGHTS
# -------------------------------

print("Computing IDF weights...")

N = len(final_tags)

df = Counter()
for tags in final_tags.values():
    df.update(set(tags))

idf = {}
for tag in tag_idx:
    idf[tag] = np.log((N + 1) / (df[tag] + 1)) + 1

# -------------------------------
# 6️⃣ BUILD IMAGE–TAG PAIRS
# -------------------------------

pairs = []

for idx, tags in final_tags.items():

    img_path = os.path.join(IMG_DIR, f"im{idx}.jpg")

    if os.path.exists(img_path):

        pairs.append((img_path, tags))

print("Total valid pairs:", len(pairs))


# -------------------------------
# 7️⃣ SPLIT DATASET
# -------------------------------

print("Splitting dataset...")
random.seed(42)
random.shuffle(pairs)

query_pairs = pairs[:QUERY_SIZE]

retrieval_pairs = pairs[QUERY_SIZE:QUERY_SIZE + RETRIEVAL_SIZE]

train_pairs = random.sample(retrieval_pairs, TRAIN_SIZE)

print("Query:", len(query_pairs))
print("Retrieval:", len(retrieval_pairs))
print("Train:", len(train_pairs))


# -------------------------------
# 8️⃣ BUILD TAG FEATURE
# -------------------------------

def build_tag_vec(tags):

    vec = np.zeros(DIM_TXT)

    tag_freq = Counter(tags)
    total_tags = len(tags)

    for t, count in tag_freq.items():
        if t in tag_idx:
            tf = count / total_tags
            vec[tag_idx[t]] = tf * idf[t]

    return vec

# -------------------------------
# 9️⃣ LOAD ALEXNET
# -------------------------------

print("Loading AlexNet...")

alexnet = models.alexnet(pretrained=True)

alexnet.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])

alexnet = alexnet.to(DEVICE)

alexnet.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])


# -------------------------------
# 🔟 FEATURE EXTRACTION
# -------------------------------

def extract_features(pair_list):

    I = []
    Tvec = []
    L = []

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

            for i,(id_,tags_) in enumerate(meta):

                I.append(feat[i])

                Tvec.append(build_tag_vec(tags_))

                L.append(image_labels[id_])

            batch_imgs = []
            meta = []

    if batch_imgs:

        batch = torch.stack(batch_imgs).to(DEVICE)

        with torch.no_grad():

            feat = alexnet(batch).cpu().numpy()

        for i,(id_,tags_) in enumerate(meta):

            I.append(feat[i])

            Tvec.append(build_tag_vec(tags_))

            L.append(image_labels[id_])

    return np.array(I), np.array(Tvec), np.array(L)


# -------------------------------
# 1️⃣1️⃣ EXTRACT DATASETS
# -------------------------------

print("Extracting QUERY features...")

I_te, T_te, L_te = extract_features(query_pairs)

print("Extracting DATABASE features...")

I_db, T_db, L_db = extract_features(retrieval_pairs)

print("Extracting TRAIN features...")

I_tr, T_tr, L_tr = extract_features(train_pairs)


# -------------------------------
# 1️⃣2️⃣ SAVE DATASETS
# -------------------------------

print("Saving .mat files...")

sio.savemat(os.path.join(OUT_DIR, "mir_query.mat"), {
    "I_te": I_te,
    "T_te": T_te,
    "L_te": L_te
})

sio.savemat(os.path.join(OUT_DIR, "mir_database.mat"), {
    "I_db": I_db,
    "T_db": T_db,
    "L_db": L_db
})

sio.savemat(os.path.join(OUT_DIR, "mir_train.mat"), {
    "I_tr": I_tr,
    "T_tr": T_tr,
    "L_tr": L_tr
})

print("Preprocessing finished successfully.")