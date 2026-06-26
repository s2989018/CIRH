import os
import json
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# ---------------- CONFIG ----------------

ROOT = "/home/s2989018/master/multilevel_chan/multilevel/datasets/NUSWIDE"
OUT_DIR = "./datasets/NUSWIDE"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# ---------------- LOAD JSON ----------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

query = load_json(os.path.join(ROOT, "query.json"))
database = load_json(os.path.join(ROOT, "database.json"))
train = load_json(os.path.join(ROOT, "train.json"))

vocab_obj = load_json(os.path.join(ROOT, "vocab.json"))

if "token_to_id" in vocab_obj:
    token_to_id = vocab_obj["token_to_id"]
else:
    token_to_id = vocab_obj

DIM_TXT = len(token_to_id)

print("Query:", len(query))
print("Database:", len(database))
print("Train:", len(train))
print("Text dim:", DIM_TXT)

# ---------------- TEXT FEATURE ----------------

def tokenize(text):
    return text.lower().split()

def build_tag_vec(sample):
    vec = np.zeros(DIM_TXT, dtype=np.float32)

    if "tags" in sample:
        tokens = sample["tags"]
    else:
        tokens = tokenize(sample["text"])

    for t in tokens:
        if t in token_to_id:
            vec[token_to_id[t]] = 1.0
        elif "[UNK]" in token_to_id:
            vec[token_to_id["[UNK]"]] = 1.0

    return vec

# ---------------- IMAGE MODEL ----------------

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

# ---------------- FEATURE EXTRACTION ----------------

def extract_features(samples):
    I = []
    Tvec = []
    L = []
    IDs = []

    batch_imgs = []
    meta = []

    for sample in tqdm(samples):
        img_path = sample["image_path"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        batch_imgs.append(img)
        meta.append(sample)

        if len(batch_imgs) == BATCH_SIZE:
            batch = torch.stack(batch_imgs).to(DEVICE)

            with torch.no_grad():
                feat = alexnet(batch).cpu().numpy()

            for i, s in enumerate(meta):
                I.append(feat[i])
                Tvec.append(build_tag_vec(s))
                L.append(np.array(s["label"], dtype=np.float32))
                IDs.append(s["photo_id"])

            batch_imgs = []
            meta = []

    if batch_imgs:
        batch = torch.stack(batch_imgs).to(DEVICE)

        with torch.no_grad():
            feat = alexnet(batch).cpu().numpy()

        for i, s in enumerate(meta):
            I.append(feat[i])
            Tvec.append(build_tag_vec(s))
            L.append(np.array(s["label"], dtype=np.float32))
            IDs.append(s["photo_id"])

    return (
        np.array(I, dtype=np.float32),
        np.array(Tvec, dtype=np.float32),
        np.array(L, dtype=np.float32),
        np.array(IDs)
    )

# ---------------- EXTRACT ----------------

print("Extracting QUERY features...")
I_te, T_te, L_te, ID_te = extract_features(query)

print("Extracting DATABASE features...")
I_db, T_db, L_db, ID_db = extract_features(database)

print("Extracting TRAIN features...")
I_tr, T_tr, L_tr, ID_tr = extract_features(train)

# ---------------- SAVE ----------------

print("Saving .mat files...")

sio.savemat(os.path.join(OUT_DIR, "nus_query.mat"), {
    "I_te": I_te,
    "T_te": T_te,
    "L_te": L_te,
    "ID_te": ID_te
})

sio.savemat(os.path.join(OUT_DIR, "nus_database.mat"), {
    "I_db": I_db,
    "T_db": T_db,
    "L_db": L_db,
    "ID_db": ID_db
})

sio.savemat(os.path.join(OUT_DIR, "nus_train.mat"), {
    "I_tr": I_tr,
    "T_tr": T_tr,
    "L_tr": L_tr,
    "ID_tr": ID_tr
})

print("Done.")
print("NUS-WIDE CIRH preprocessing finished.")