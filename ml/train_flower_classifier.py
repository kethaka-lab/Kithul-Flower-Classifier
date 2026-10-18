"""
Palm flower classifier: kithul / coconut / date (add more class folders any time).

Expected dataset layout:
    data/
      kithul/    *.jpg
      coconut/   *.jpg   (include cut-surface images here too)
      date/      *.jpg

Install:  pip install torch torchvision scikit-learn matplotlib
Run:      python train_flower_classifier.py --data data --epochs 25
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--img", type=int, default=224)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--out", default="flower_model.pt")
args = p.parse_args()

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(args.img, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tf = transforms.Compose([
    transforms.Resize(int(args.img * 1.14)),
    transforms.CenterCrop(args.img),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Two views of the same folder so train/val get different transforms
full_train = datasets.ImageFolder(args.data, transform=train_tf)
full_eval = datasets.ImageFolder(args.data, transform=eval_tf)
classes = full_train.classes
labels = np.array(full_train.targets)
print("Classes:", classes, "| counts:", np.bincount(labels))

# Stratified 70/15/15 split
idx = np.arange(len(labels))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=labels[tmp_idx], random_state=SEED)

# Balance classes in training (coconut data is likely to be the smallest)
class_counts = np.bincount(labels[tr_idx])
weights = (1.0 / class_counts)[labels[tr_idx]]
sampler = WeightedRandomSampler(weights, num_samples=len(tr_idx), replacement=True)

train_dl = DataLoader(Subset(full_train, tr_idx), batch_size=args.batch, sampler=sampler, num_workers=2)
val_dl = DataLoader(Subset(full_eval, va_idx), batch_size=args.batch, num_workers=2)
test_dl = DataLoader(Subset(full_eval, te_idx), batch_size=args.batch, num_workers=2)

# Pretrained backbone, new head
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.to(device)

def set_backbone_trainable(flag):
    for prm in model.features.parameters():
        prm.requires_grad = flag

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def run_epoch(dl, opt=None):
    train = opt is not None
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, correct / n

best_acc, patience, bad = 0.0, 6, 0
warmup = min(5, args.epochs)

for epoch in range(args.epochs):
    if epoch == 0:  # phase 1: train head only
        set_backbone_trainable(False)
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    elif epoch == warmup:  # phase 2: fine-tune everything, lower LR
        set_backbone_trainable(True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 20)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - warmup)
    tl, ta = run_epoch(train_dl, opt)
    vl, va = run_epoch(val_dl)
    if epoch >= warmup:
        sched.step()
    print(f"epoch {epoch+1:02d} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
    if va > best_acc:
        best_acc, bad = va, 0
        torch.save({"state_dict": model.state_dict(), "classes": classes, "img": args.img}, args.out)
    else:
        bad += 1
        if bad >= patience and epoch >= warmup:
            print("Early stopping"); break

# Final evaluation on the untouched test split
ckpt = torch.load(args.out, map_location=device)
model.load_state_dict(ckpt["state_dict"]); model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_dl:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
print("\nTest results")
print(classification_report(y_true, y_pred, target_names=classes, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
json.dump(classes, open("classes.json", "w"))
