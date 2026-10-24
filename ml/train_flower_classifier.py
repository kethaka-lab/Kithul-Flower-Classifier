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
p.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "resnet50"],
               help="Model architecture: efficientnet_b0 | resnet18 | resnet50")
p.add_argument("--splits", default="ml/splits.json", help="Path to stratified train/val/test split JSON")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--img", type=int, default=224)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--out", default=None, help="Output weights file (defaults to ml/weights/<model_name>.pt)")
args = p.parse_args()

if args.out is None:
    args.out = f"ml/weights/{args.model}.pt"

Path(args.out).parent.mkdir(parents=True, exist_ok=True)

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

# Custom dataset to only load true flower classes (coconut, date, kithul) and ignore evaluation folders like data/test
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension, IMG_EXTENSIONS

class PalmFlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.loader = default_loader
        # Only include actual flower class subdirectories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir() and d.name.lower() != "test"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            cdir = self.root_dir / c
            for f in sorted(cdir.iterdir()):
                if f.is_file() and has_file_allowed_extension(f.name, IMG_EXTENSIONS):
                    self.samples.append((str(f), self.class_to_idx[c]))
        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Two views of the same dataset so train/val get different transforms
full_train = PalmFlowerDataset(args.data, transform=train_tf)
full_eval = PalmFlowerDataset(args.data, transform=eval_tf)
classes = full_train.classes
labels = np.array(full_train.targets)
print("Classes:", classes, "| counts:", np.bincount(labels))

# Reproducible split persistence
splits_file = Path(args.splits)
all_samples = full_train.samples  # list of (filepath, class_idx)
path_to_idx = {Path(s[0]).resolve().as_posix(): idx for idx, s in enumerate(all_samples)}

if splits_file.exists():
    print(f"Loading existing split from {splits_file}...")
    with open(splits_file, "r") as f:
        splits_data = json.load(f)
    
    def resolve_indices(paths):
        resolved = []
        for p_str in paths:
            norm_p = Path(p_str).resolve().as_posix()
            if norm_p not in path_to_idx:
                raise ValueError(f"File from {splits_file} not found in dataset '{args.data}': {p_str}")
            resolved.append(path_to_idx[norm_p])
        return resolved

    tr_idx = resolve_indices(splits_data["train"])
    va_idx = resolve_indices(splits_data["val"])
    te_idx = resolve_indices(splits_data["test"])
else:
    print(f"Generating new reproducible 70/15/15 stratified split and saving to {splits_file}...")
    idx = np.arange(len(labels))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=labels[tmp_idx], random_state=SEED)

    tr_idx = tr_idx.tolist()
    va_idx = va_idx.tolist()
    te_idx = te_idx.tolist()

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    splits_dict = {
        "classes": classes,
        "seed": SEED,
        "train": [Path(all_samples[i][0]).as_posix() for i in tr_idx],
        "val": [Path(all_samples[i][0]).as_posix() for i in va_idx],
        "test": [Path(all_samples[i][0]).as_posix() for i in te_idx],
    }
    with open(splits_file, "w") as f:
        json.dump(splits_dict, f, indent=2)

print(f"Split size - Train: {len(tr_idx)}, Val: {len(va_idx)}, Test: {len(te_idx)}")

# Balance classes in training (coconut data is likely to be the smallest)
class_counts = np.bincount(labels[tr_idx])
weights = (1.0 / class_counts)[labels[tr_idx]]
sampler = WeightedRandomSampler(weights, num_samples=len(tr_idx), replacement=True)

train_dl = DataLoader(Subset(full_train, tr_idx), batch_size=args.batch, sampler=sampler, num_workers=0)
val_dl = DataLoader(Subset(full_eval, va_idx), batch_size=args.batch, num_workers=0)
test_dl = DataLoader(Subset(full_eval, te_idx), batch_size=args.batch, num_workers=0)

# Pretrained backbone, new head via timm
import timm

MODEL_NAME_MAP = {
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
}

timm_model_name = MODEL_NAME_MAP[args.model]
print(f"Initializing {args.model} ({timm_model_name}) with pretrained weights...")
model = timm.create_model(timm_model_name, pretrained=True, num_classes=len(classes))

# Initialize classifier head
clf = model.get_classifier()
nn.init.normal_(clf.weight, std=0.01)
if clf.bias is not None:
    nn.init.zeros_(clf.bias)
model.to(device)

clf_param_ids = {id(p) for p in clf.parameters()}

def set_backbone_trainable(flag):
    for prm in model.parameters():
        if id(prm) not in clf_param_ids:
            prm.requires_grad = flag

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def run_epoch(dl, opt=None):
    train = opt is not None
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    all_y, all_preds = [], []
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            n += len(y)
            all_y.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(all_y, all_preds, average="macro", zero_division=0))
    return loss_sum / n, correct / n, macro_f1

best_f1, best_loss, patience, bad = -1.0, float("inf"), 6, 0
saved_epoch = -1
warmup = min(5, args.epochs)
phase2_min_epochs = 8

log_dir = Path("ml/results")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"train_{args.model}.log"
log_lines = []
header = f"Training log for {args.model} (epochs={args.epochs}, lr={args.lr}, batch={args.batch})"
print(header)
log_lines.append(header)
log_lines.append("epoch,phase,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1,saved")

for epoch in range(args.epochs):
    phase = 1 if epoch < warmup else 2
    if epoch == 0:  # phase 1: train head only
        set_backbone_trainable(False)
        opt = torch.optim.AdamW(clf.parameters(), lr=args.lr)
    elif epoch == warmup:  # phase 2: fine-tune everything with discriminative learning rates
        set_backbone_trainable(True)
        backbone_params = [p for p in model.parameters() if id(p) not in clf_param_ids]
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr / 20},
            {"params": clf.parameters(), "lr": args.lr},
        ])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - warmup)
        # Reset patience counter for Phase 2
        bad = 0

    tl, ta, tf1 = run_epoch(train_dl, opt)
    vl, va, vf1 = run_epoch(val_dl)
    if epoch >= warmup:
        sched.step()

    epoch_saved = False
    if vf1 > best_f1 or (abs(vf1 - best_f1) < 1e-6 and vl < best_loss):
        best_f1, best_loss, bad = vf1, vl, 0
        saved_epoch = epoch + 1
        epoch_saved = True
        torch.save({
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "classes": classes,
            "img": args.img,
            "epoch": saved_epoch,
            "phase": phase,
            "val_f1": best_f1,
            "val_loss": best_loss,
            "val_acc": va,
        }, args.out)
    else:
        bad += 1
        # Early stopping condition: must be in Phase 2 and have completed at least 8 epochs of Phase 2
        phase2_epochs_run = (epoch + 1) - warmup
        if bad >= patience and epoch >= warmup and phase2_epochs_run >= phase2_min_epochs:
            print(f"Early stopping at epoch {epoch+1:02d} (Phase 2 ran {phase2_epochs_run} epochs)")
            break

    status_str = f"epoch {epoch+1:02d} (P{phase}) | train loss {tl:.3f} acc {ta:.3f} f1 {tf1:.3f} | val loss {vl:.3f} acc {va:.3f} f1 {vf1:.3f}{' [SAVED]' if epoch_saved else ''}"
    print(status_str)
    log_lines.append(f"{epoch+1},{phase},{tl:.4f},{ta:.4f},{tf1:.4f},{vl:.4f},{va:.4f},{vf1:.4f},{epoch_saved}")

with open(log_file, "w") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"\nTraining completed. Saved checkpoint is from Epoch {saved_epoch} (val macro-F1: {best_f1:.3f}, val loss: {best_loss:.3f}).")
if saved_epoch <= warmup:
    print(f"⚠️  WARNING: Saved checkpoint is from Epoch {saved_epoch} (frozen warm-up Phase 1)! The unfreeze phase did not yield a better validation score.")

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
