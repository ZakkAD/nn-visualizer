#!/usr/bin/env python3
"""
print_fail_viz.py - train a CNN to classify 3D-print failures from camera
images, with a live full-colour terminal dashboard.

    ./print_fail_viz.py                         # Kaggle "3d printing errors" dataset (auto-download)
    ./print_fail_viz.py --size 160 --epochs 30  # bigger images, longer training (more accurate, slower)
    ./print_fail_viz.py --arch small            # tiny CNN from scratch: fast, less accurate
    ./print_fail_viz.py --report-only           # rebuild the HTML review for models/print_fail_best.pt
    ./print_fail_viz.py --demo                  # synthetic print images, no download
    ./print_fail_viz.py --data path/to/dataset  # your own images (class folders, or a Kaggle zip)

Model: ResNet-18 pretrained on ImageNet and fine-tuned (transfer learning),
128px images, zoom/flip/colour augmentation, predictions averaged with a
mirrored copy. The best epoch is chosen by balanced accuracy (mean recall
over labels), so the rare spaghetti class counts as much as the others.

After training, an HTML review report is written to reports/<date>/:
index.html (metrics, clickable confusion matrix) plus one page per label
showing misses, false alarms and correct examples with Grad-CAM heatmaps.

Default dataset: https://www.kaggle.com/datasets/nimbus200/3d-printing-errors
(good / under-extrusion / stringing / spaghetti, 34 printed shapes). Validation
uses whole shapes the network never saw in training, so the score reflects
how it does on new prints rather than on near-duplicate frames.

The best model (by balanced validation accuracy) and the final model are
saved to models/ next to this script.

Own-dataset layout (folder names become the class names):

    dataset/                      or     dataset/
      good/      img1.jpg ...              train/good/ ...  train/spaghetti/ ...
      spaghetti/ ...                       val/good/ ...    val/spaghetti/ ...
      stringing/ ...
      layer_shift/ ...

Without a train/val split, 15% of each class is held out for validation.

Dashboard: the augmented input the network sees, a Grad-CAM heatmap of where
it is looking, the most active feature maps of the first three conv stages,
class probabilities, loss/accuracy curves and a validation confusion matrix.

Needs: Python 3.8+, PyTorch, torchvision, NumPy, Pillow (run ./install.sh), and a
truecolor terminal of at least 122x43 characters.
"""
import argparse
import csv
import math
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor


def _use_local_venv():
    """If a .venv sits next to this script and we're not already in it, re-exec inside it."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, ".venv")
    py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python")
    if os.path.exists(py) and os.path.realpath(sys.prefix) != os.path.realpath(venv) \
            and not os.environ.get("MNIST_VIZ_NO_VENV"):
        os.environ["MNIST_VIZ_NO_VENV"] = "1"
        os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:])


if __name__ == "__main__":
    _use_local_venv()

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as e:
    sys.exit(f"Missing package '{e.name}'. Run ./install.sh first (it creates a .venv with everything needed).")


HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _has_images(d):
    return any(os.path.splitext(f)[1].lower() in IMG_EXT for f in os.listdir(d))


def _class_dirs(root):
    return sorted(c for c in os.listdir(root)
                  if os.path.isdir(os.path.join(root, c)) and _has_images(os.path.join(root, c)))


def _load_split(root, classes, size):
    xs, ys, ps = [], [], []
    for ci, c in enumerate(classes):
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)
        for k, fn in enumerate(files):
            try:
                im = Image.open(os.path.join(d, fn))
                im.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale
                im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            except Exception as e:
                print(f"  skipping {fn}: {e}")
                continue
            xs.append(np.asarray(im, dtype=np.uint8))
            ys.append(ci)
            ps.append(os.path.join(d, fn))
            if k % 200 == 0:
                print(f"\r  {c:<20} {k + 1}/{len(files)}", end="", flush=True)
        print(f"\r  {c:<20} {len(files)} images          ")
    return np.stack(xs), np.array(ys, dtype=np.int64), np.array(ps)


def load_dataset(root, size, val_frac, rng):
    if not os.path.isdir(root):
        sys.exit(f"Dataset folder '{root}' not found. Use --data PATH, or --demo for synthetic images.")
    val_name = next((v for v in ("val", "valid", "validation", "test") if os.path.isdir(os.path.join(root, v))), None)
    if os.path.isdir(os.path.join(root, "train")) and val_name:
        classes = _class_dirs(os.path.join(root, "train"))
        print(f"Loading train/ ({len(classes)} classes)…")
        xtr, ytr, ptr = _load_split(os.path.join(root, "train"), classes, size)
        print(f"Loading {val_name}/…")
        xva, yva, pva = _load_split(os.path.join(root, val_name), classes, size)
        return xtr, ytr, ptr, xva, yva, pva, classes
    classes = _class_dirs(root)
    if len(classes) < 2:
        sys.exit(f"Need at least 2 class folders with images inside '{root}'.")
    print(f"Loading {len(classes)} classes from {root} (holding out {val_frac:.0%} for validation)…")
    x, y, p = _load_split(root, classes, size)
    mask = stratified_split(y, val_frac, rng)
    return x[~mask], y[~mask], p[~mask], x[mask], y[mask], p[mask], classes


# -- Kaggle "3d printing errors" dataset ----------------------------------------
KAGGLE_ID = "nimbus200/3d-printing-errors"
KAGGLE_URL = "https://www.kaggle.com/datasets/" + KAGGLE_ID
KAGGLE_CLASSES = {0: "good", 1: "under_extrusion", 2: "stringing", 3: "class_3", 4: "spaghetti"}
KAGGLE_CSV = "all_images_no_filter.csv"


def fetch_kaggle():
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is missing. Run ./install.sh (or: pip install kagglehub).")
    print(f"Fetching {KAGGLE_ID} via kagglehub (≈600 MB on first run, cached afterwards)…")
    try:
        return kagglehub.dataset_download(KAGGLE_ID)
    except Exception as e:
        sys.exit(f"Download failed: {e}\n\n"
                 f"Either add a Kaggle API token (kaggle.com → Settings → API → Create New Token,\n"
                 f"save it as ~/.kaggle/kaggle.json), or download the zip by hand from\n  {KAGGLE_URL}\n"
                 f"and run:  ./print_fail_viz.py --data path/to/archive.zip")


def find_file(root, name):
    for dp, _, files in os.walk(root):
        if name in files:
            return os.path.join(dp, name)
    return None


def is_kaggle_layout(root):
    return find_file(root, KAGGLE_CSV) is not None or find_file(root, "all_images.csv") is not None


def _load_one(path, size):
    try:
        im = Image.open(path)
        im.draft("RGB", (size * 2, size * 2))
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None


def load_images(paths, size):
    out = [None] * len(paths)
    with ThreadPoolExecutor(min(8, os.cpu_count() or 4)) as ex:
        for k, img in enumerate(ex.map(lambda q: _load_one(q, size), paths)):
            out[k] = img
            if k % 250 == 0 or k == len(paths) - 1:
                print(f"\r  loading images {k + 1:,}/{len(paths):,}", end="", flush=True)
    print()
    return out


def load_kaggle(root, csv_name, size, cache_dir):
    """Returns x, y, groups (printed shape id per image), image paths, class names. Caches resized arrays."""
    csv_path = find_file(root, csv_name)
    if csv_path is None:
        sys.exit(f"Could not find {csv_name} inside {root}.")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"printing_errors_{os.path.splitext(csv_name)[0]}_{size}px_v2.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(csv_path):
        print(f"Using cached images from {cache}")
        z = np.load(cache, allow_pickle=False)
        return z["x"], z["y"], z["groups"], z["paths"], [str(c) for c in z["classes"]]

    print(f"Indexing images under {root}…")
    index = {}
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                p = os.path.join(dp, f)
                # several resolutions may exist: prefer the 256px copies (fast to decode)
                if f not in index or ("256" in dp and "256" not in os.path.dirname(index[f])):
                    index[f] = p

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        head = fh.readline()
        delim = max(";,\t|", key=head.count)  # this dataset uses ';'
        fh.seek(0)
        rows = list(csv.DictReader(fh, delimiter=delim))
    if not rows:
        sys.exit(f"{csv_path} is empty.")
    cols = {c.strip().lower(): c for c in rows[0].keys() if c}
    img_col = next((cols[c] for c in ("image", "img", "filename", "file", "image_name", "path") if c in cols),
                   next((v for k, v in cols.items() if "image" in k or "file" in k), None))
    cls_col = next((cols[c] for c in ("class", "label", "error", "category") if c in cols), None)
    shp_col = cols.get("shape")
    if img_col is None or cls_col is None:
        sys.exit(f"Unexpected columns in {csv_path}: {list(rows[0].keys())}")

    paths, labels, groups, missing = [], [], [], 0
    for r in rows:
        name = os.path.basename(str(r[img_col]).strip().replace("\\", "/"))
        p = index.get(name) or next((index.get(name + e) for e in (".png", ".jpg") if name + e in index), None)
        if p is None:
            missing += 1
            continue
        try:
            k = int(float(r[cls_col]))
            g = int(float(r[shp_col])) if shp_col and str(r[shp_col]).strip() else -1
        except ValueError:
            missing += 1
            continue
        paths.append(p)
        labels.append(k)
        groups.append(g)
    if missing:
        print(f"  {missing:,} CSV rows had no matching image and were skipped")
    if not paths:
        sys.exit("No images matched the CSV.")

    ids = sorted(set(labels))
    remap = {k: i for i, k in enumerate(ids)}
    classes = [KAGGLE_CLASSES.get(k, f"class_{k}") for k in ids]
    imgs = load_images(paths, size)
    keep = [i for i, im in enumerate(imgs) if im is not None]
    x = np.stack([imgs[i] for i in keep])
    y = np.array([remap[labels[i]] for i in keep], dtype=np.int64)
    g = np.array([groups[i] for i in keep], dtype=np.int64)
    pa = np.array([paths[i] for i in keep])
    np.savez(cache, x=x, y=y, groups=g, paths=pa, classes=np.array(classes))
    print(f"  cached to {cache}")
    return x, y, g, pa, classes


def stratified_split(y, val_frac, rng):
    mask = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.nonzero(y == c)[0])
        mask[idx[:max(1, int(len(idx) * val_frac))]] = True
    return mask


def group_split(y, groups, val_frac, rng):
    """Hold out whole printed shapes until every class has ~val_frac of its images in validation."""
    nc = int(y.max()) + 1
    if (groups < 0).all():
        return None
    target = val_frac * np.bincount(y, minlength=nc)
    have = np.zeros(nc)
    val_groups = []
    for g in rng.permutation(np.unique(groups[groups >= 0])):
        cnt = np.bincount(y[groups == g], minlength=nc)
        if np.any((have < target) & (cnt > 0)):
            val_groups.append(g)
            have += cnt
        if np.all(have >= target):
            break
    mask = np.isin(groups, val_groups)
    if (np.bincount(y[~mask], minlength=nc) == 0).any() or (np.bincount(y[mask], minlength=nc) == 0).any():
        return None
    return mask


def extract_zip(path, dest_root):
    dest = os.path.join(dest_root, os.path.splitext(os.path.basename(path))[0])
    if not os.path.isdir(dest):
        print(f"Extracting {path} → {dest}…")
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    return dest


# -- synthetic demo data ------------------------------------------------------
DEMO_CLASSES = ["good", "layer_shift", "spaghetti", "stringing", "warping"]
FILAMENTS = [(220, 40, 40), (240, 240, 240), (30, 30, 30), (40, 120, 230), (250, 170, 30),
             (60, 200, 90), (200, 60, 200), (240, 220, 60), (150, 150, 160)]


def demo_image(cls, rng, size):
    S = 128
    yy = np.linspace(0, 1, S)[:, None, None]
    plate = rng.integers(30, 80)
    arr = np.clip(plate + 30 * yy + rng.normal(0, 4, (S, S, 3)), 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, S, rng.integers(20, 35)], fill=tuple(int(v) for v in rng.integers(5, 35, 3)))
    col = FILAMENTS[rng.integers(len(FILAMENTS))]
    dark = tuple(int(v * 0.72) for v in col)
    base, top = int(rng.integers(100, 116)), int(rng.integers(35, 60))
    cx, half = int(rng.integers(42, 86)), int(rng.integers(14, 26))

    def tower(x0, x1, y0, y1, shift_at=None, shift=0, warp=0):
        for y in range(y1, y0, -2):
            off = shift if (shift_at is not None and y < shift_at) else 0
            lift = 0
            if warp and y > y1 - 14:
                lift = int(warp * ((y - (y1 - 14)) / 14) ** 2)
            d.line([x0 + off + lift, y, x1 + off - lift, y], fill=col if (y // 2) % 2 else dark, width=2)

    if cls == "good":
        tower(cx - half, cx + half, top, base)
    elif cls == "layer_shift":
        tower(cx - half, cx + half, top, base, shift_at=int(rng.integers(top + 12, base - 15)),
              shift=int(rng.choice([-1, 1]) * rng.integers(7, 18)))
    elif cls == "warping":
        w = int(rng.integers(6, 12))
        tower(cx - half, cx + half, top, base, warp=w)
        for sx in (cx - half, cx + half - w):
            d.polygon([(sx, base + 1), (sx + w, base + 1), (sx + (w if sx < cx else 0), base - 10)],
                      fill=(10, 10, 12))
    elif cls == "stringing":
        g = int(rng.integers(18, 30))
        hw = max(6, half // 2)
        tower(cx - g - hw, cx - g + hw, top, base)
        tower(cx + g - hw, cx + g + hw, top, base)
        for _ in range(int(rng.integers(5, 12))):
            y0 = int(rng.integers(top, base - 5))
            sag = int(rng.integers(2, 10))
            pts = [(cx - g + hw + t, y0 + sag * math.sin(math.pi * t / (2 * (g - hw)))) for t in range(0, 2 * (g - hw) + 1, 3)]
            d.line(pts, fill=col, width=1)
    elif cls == "spaghetti":
        h = int(rng.integers(5, 25))
        tower(cx - half, cx + half, base - h, base)
        for _ in range(int(rng.integers(3, 7))):
            x, y = cx + rng.integers(-half, half), base - h - rng.integers(0, 10)
            pts = [(x, y)]
            ang = rng.uniform(0, 2 * math.pi)
            for _ in range(int(rng.integers(25, 60))):
                ang += rng.normal(0, 0.7)
                x = float(np.clip(x + 4 * math.cos(ang), 10, S - 10))
                y = float(np.clip(y + 4 * math.sin(ang), 30, base + 6))
                pts.append((x, y))
            d.line(pts, fill=col if rng.random() < 0.6 else dark, width=2)
    im = im.filter(ImageFilter.GaussianBlur(0.6)).resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) * rng.uniform(0.75, 1.2) + rng.normal(0, 5, (size, size, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def demo_dataset(size, rng, n_train=320, n_val=60):
    print(f"Generating synthetic print images ({len(DEMO_CLASSES)} classes)…")
    out = []
    for n in (n_train, n_val):
        xs, ys = [], []
        for ci, c in enumerate(DEMO_CLASSES):
            for _ in range(n):
                xs.append(demo_image(c, rng, size))
                ys.append(ci)
        out += [np.stack(xs), np.array(ys, dtype=np.int64)]
    return (*out, list(DEMO_CLASSES))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class PrintCNN(nn.Module):
    """Small CNN trained from scratch: 4 conv stages (w..8w channels) -> global avg pool -> linear."""
    mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    cam_index = 3

    def __init__(self, n_classes, width=16):
        super().__init__()
        chans = [3, width, width * 2, width * 4, width * 8]
        self.stages = nn.ModuleList(conv_block(a, b) for a, b in zip(chans[:-1], chans[1:]))
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(chans[-1], n_classes))

    def forward(self, x):
        feats = []
        for i, s in enumerate(self.stages):
            x = s(x)
            feats.append(x)
            if i < len(self.stages) - 1:
                x = self.pool(x)
        return self.head(x.mean((2, 3))), feats


class ResNetCNN(nn.Module):
    """ResNet-18 pretrained on ImageNet, fine-tuned. Exposes the 4 stage outputs for the dashboard."""
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    cam_index = 2  # layer3: 8x8 at 128px, finer than layer4's 4x4

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        try:
            import torchvision
        except ImportError:
            sys.exit("torchvision is missing. Run ./install.sh again.")
        m = None
        if pretrained:
            try:
                print("Loading ImageNet-pretrained ResNet-18 weights (≈45 MB download on first run)…")
                m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            except Exception as e:
                print(f"  could not get pretrained weights ({e}); training from scratch instead")
        if m is None:
            m = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.stages = nn.ModuleList([m.layer1, m.layer2, m.layer3, m.layer4])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        return self.head(x.mean((2, 3))), feats


ARCH_DEFAULTS = {  # size, epochs, lr
    "resnet18": (128, 20, 1e-3),
    "small": (64, 25, 2e-3),
}


def make_model(arch, n_classes, width=16, pretrained=True):
    if arch == "resnet18":
        return ResNetCNN(n_classes, pretrained)
    return PrintCNN(n_classes, width)


_NORM = {}


def set_norm(model, dev):
    _NORM["m"] = torch.tensor(model.mean, device=dev).view(1, 3, 1, 1)
    _NORM["s"] = torch.tensor(model.std, device=dev).view(1, 3, 1, 1)


def normalize(x01):
    return (x01 - _NORM["m"]) / _NORM["s"]


def augment(x):
    """x: (B,3,S,S) in [0,1]. Random zoom-crop, h-flip, brightness / contrast / saturation jitter."""
    B, _, S, _ = x.shape
    dv = x.device
    cs = int(S * np.random.uniform(0.72, 1.0))
    if cs < S:
        i, j = np.random.randint(0, S - cs + 1, 2)
        x = F.interpolate(x[:, :, i:i + cs, j:j + cs], size=(S, S), mode="bilinear", align_corners=False)
    flip = torch.rand(B, 1, 1, 1, device=dv) < 0.5
    x = torch.where(flip, x.flip(3), x)

    def r(lo, hi):
        return lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=dv)

    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * r(0.6, 1.4)                                 # saturation
    m = x.mean((1, 2, 3), keepdim=True)
    x = m + (x - m) * r(0.7, 1.3)                                       # contrast
    x = x * r(0.75, 1.25)                                               # brightness
    return x.clamp(0, 1)


def to_tensor(x_uint8, dev):
    return torch.from_numpy(np.ascontiguousarray(x_uint8)).to(dev).permute(0, 3, 1, 2).float().div_(255)


def gradcam_batch(model, x01):
    """Predictions (averaged with a mirrored copy) + Grad-CAM for the predicted class. Model must be in eval()."""
    xn = normalize(x01)
    with torch.no_grad():
        lflip, _ = model(xn.flip(3))
    with torch.enable_grad():
        logits, feats = model(xn)
        A = feats[model.cam_index]
        probs = F.softmax((logits.detach() + lflip) / 2, 1)
        pred = probs.argmax(1)
        g, = torch.autograd.grad(logits.gather(1, pred[:, None]).sum(), A)
    cam = F.relu((g.mean((2, 3), keepdim=True) * A.detach()).sum(1, keepdim=True))
    return probs, cam, feats


def snapshot(model, x01, disp=32):
    """Everything the dashboard shows for one image: probs, Grad-CAM, top feature maps."""
    was_training = model.training
    model.eval()
    probs, cam, feats = gradcam_batch(model, x01[None])
    model.train(was_training)
    cam = F.interpolate(cam, size=(disp, disp), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    img = F.interpolate(x01[None], size=(disp, disp), mode="area")[0]
    img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    maps = []
    for f in feats[:3]:
        f = f[0].detach()
        top = f.mean((1, 2)).topk(min(5, f.shape[0])).indices
        m = F.adaptive_avg_pool2d(f[top][None], 8)[0]
        m = m / (m.amax((1, 2), keepdim=True) + 1e-6)
        maps.append((m.cpu().numpy(), tuple(f.shape)))
    return dict(img=img, cam=cam, probs=probs[0].cpu().numpy(), maps=maps)


# ----------------------------------------------------------------------------
# Terminal drawing primitives
# ----------------------------------------------------------------------------
RESET = "\x1b[0m"


def fg(c):
    return f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m"


def bg(c):
    return f"\x1b[48;2;{c[0]};{c[1]};{c[2]}m"


def mix(a, b, t):
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))


def grad(stops, t):
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return mix(stops[i], stops[i + 1], t - i)


THEMES = {
    "neon": dict(seq=[(8, 8, 20), (40, 20, 90), (150, 30, 190), (255, 60, 170), (0, 220, 255), (235, 255, 255)],
                 neg=(0, 160, 255), pos=(255, 50, 160), accent=(0, 230, 255), accent2=(255, 80, 200),
                 dim=(70, 70, 105), text=(205, 205, 235), good=(80, 255, 160), bad=(255, 80, 90)),
    "crimson": dict(seq=[(10, 0, 0), (60, 0, 6), (140, 6, 20), (220, 20, 40), (255, 90, 80), (255, 225, 215)],
                    neg=(120, 120, 140), pos=(255, 30, 50), accent=(255, 45, 65), accent2=(255, 160, 160),
                    dim=(105, 35, 40), text=(235, 205, 205), good=(255, 215, 210), bad=(255, 30, 30)),
    "matrix": dict(seq=[(0, 8, 0), (0, 50, 10), (0, 120, 30), (40, 200, 60), (150, 255, 150), (230, 255, 230)],
                   neg=(0, 110, 90), pos=(170, 255, 90), accent=(60, 255, 100), accent2=(190, 255, 190),
                   dim=(20, 90, 35), text=(170, 235, 170), good=(120, 255, 140), bad=(255, 90, 60)),
}


class Theme:
    def __init__(self, d):
        self.__dict__.update(d)
        self.seq_rgb = [grad(d["seq"], i / 255) for i in range(256)]
        self.F = [fg(c) for c in self.seq_rgb]
        self.B = [bg(c) for c in self.seq_rgb]
        self.fa, self.fa2 = fg(d["accent"]), fg(d["accent2"])
        self.fd, self.ft = fg(d["dim"]), fg(d["text"])
        self.fgood, self.fbad = fg(d["good"]), fg(d["bad"])


def gtext(text, th, phase=0.0, lo=0.5, hi=1.0):
    n = max(len(text), 1)
    return "".join(th.F[int((lo + (hi - lo) * (0.5 + 0.5 * math.sin(phase + 2 * math.pi * k / n))) * 255)] + ch
                   for k, ch in enumerate(text)) + RESET


def halfblock(idx, FL, BL):
    """idx: (2R, C) palette indices -> R lines of C chars."""
    return ["".join(FL[a] + BL[b] + "▀" for a, b in zip(idx[r], idx[r + 1])) + RESET
            for r in range(0, idx.shape[0], 2)]


def halfblock_rgb(img):
    """img: (2R, C, 3) uint8 -> R lines of C chars in true colour."""
    return ["".join(fg(a) + bg(b) + "▀" for a, b in zip(img[r], img[r + 1])) + RESET
            for r in range(0, img.shape[0], 2)]


BLOCKS = " ▏▎▍▌▋▊▉"


def cbar(frac, width, th, lo=0.35):
    frac = min(max(frac, 0.0), 1.0)
    total = frac * width
    full, rem = int(total), int((total - int(total)) * 8)
    out = []
    for k in range(width):
        col = th.F[int((lo + (1 - lo) * k / max(width - 1, 1)) * 255)]
        if k < full:
            out.append(col + "█")
        elif k == full and rem:
            out.append(col + BLOCKS[rem])
        else:
            out.append(th.fd + "·")
    return "".join(out) + RESET


def pbar(frac, width, th, phase):
    frac = min(max(frac, 0.0), 1.0)
    n = int(frac * width)
    out = []
    for k in range(width):
        if k < n:
            out.append(th.F[int((0.55 + 0.45 * (0.5 + 0.5 * math.sin(phase * 3 - k * 0.25))) * 255)] + "━")
        elif k == n:
            out.append(th.fa + "╸")
        else:
            out.append(th.fd + "─")
    return "".join(out) + RESET


DOT = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], np.uint8)


class Braille:
    """w x h braille cells = (2w x 4h) sub-pixels. Positive values: main series, negative: second series."""

    def __init__(self, w, h):
        self.w, self.h, self.pw, self.ph = w, h, 2 * w, 4 * h
        self.bits = np.zeros((h, w), np.uint8)
        self.pos = np.zeros((h, w), np.float32)
        self.neg = np.zeros((h, w), np.float32)

    def line(self, x0, y0, x1, y1, v):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.rint(np.linspace(x0, x1, n)).astype(int)
        ys = np.rint(np.linspace(y0, y1, n)).astype(int)
        m = (xs >= 0) & (xs < self.pw) & (ys >= 0) & (ys < self.ph)
        xs, ys = xs[m], ys[m]
        cy, cx = ys // 4, xs // 2
        np.bitwise_or.at(self.bits, (cy, cx), DOT[ys % 4, xs % 2])
        np.maximum.at(self.pos if v >= 0 else self.neg, (cy, cx), abs(v))

    def render(self, th):
        si = np.clip(self.pos * 255, 0, 255).astype(int)
        lines = []
        for r in range(self.h):
            s = []
            for c in range(self.w):
                b = self.bits[r, c]
                if not b:
                    s.append(" ")
                elif self.neg[r, c] > 0:
                    s.append(th.fa2 + chr(0x2800 + int(b)))
                else:
                    s.append(th.F[si[r, c]] + chr(0x2800 + int(b)))
            lines.append("".join(s) + RESET)
        return lines


def line_chart(series, w, h, th, ymin=None, ymax=None, extra=None):
    """series: per-step values (gradient line). extra: (step_indices, values) drawn in accent2."""
    c = Braille(w, h)
    ys = np.asarray(series, dtype=np.float64)
    n = len(ys)
    if n < 2:
        return c.render(th)
    npx = c.pw
    if n > npx:
        edges = np.linspace(0, n, npx + 1).astype(int)
        ys = np.array([ys[edges[i]:edges[i + 1]].mean() for i in range(npx)])
    ex_x, ex_y = (np.asarray(extra[0], float), np.asarray(extra[1], float)) if extra and len(extra[0]) else (None, None)
    allv = ys if ex_y is None else np.concatenate([ys, ex_y])
    lo = allv.min() if ymin is None else ymin
    hi = allv.max() if ymax is None else ymax
    span = (hi - lo) or 1.0

    def ypx(v):
        return (1 - (np.clip(v, lo, hi) - lo) / span) * (c.ph - 1)

    xs = np.linspace(0, npx - 1, len(ys))
    py = ypx(ys)
    for i in range(len(ys) - 1):
        c.line(xs[i], py[i], xs[i + 1], py[i + 1], 0.35 + 0.65 * (1 - py[i + 1] / c.ph))
    if ex_x is not None:
        ex = ex_x / max(n - 1, 1) * (npx - 1)
        ey = ypx(ex_y)
        for i in range(len(ex)):
            c.line(ex[i] - 1, ey[i], ex[i] + 1, ey[i], -1)
            if i:
                c.line(ex[i - 1], ey[i - 1], ex[i], ey[i], -1)
    return c.render(th)


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
NEED_W, NEED_H = 122, 43
MAXC = 10  # classes shown in bars / confusion matrix


class Dashboard:
    def __init__(self, classes, th, info):
        self.classes, self.th, self.info = classes, th, info
        self.last_size = None
        self.t0 = time.time()

    def box(self, P, r, c, h, w, title=""):
        th = self.th
        P(r, c, th.fd + "╭" + "─" * (w - 2) + "╮")
        for k in range(1, h - 1):
            P(r + k, c, th.fd + "│")
            P(r + k, c + w - 1, th.fd + "│")
        P(r + h - 1, c, th.fd + "╰" + "─" * (w - 2) + "╯" + RESET)
        if title:
            P(r, c + 2, th.fa + " " + title + " " + RESET)

    def cam_overlay(self, img, cam):
        th = self.th
        out = np.empty_like(img)
        lum = img.astype(np.float32).mean(2) * 0.55
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                v = float(cam[y, x])
                heat = th.seq_rgb[int(v * 255)]
                g = lum[y, x]
                a = min(1.0, v ** 0.8) * 0.9
                out[y, x] = (g + (heat[0] - g) * a, g + (heat[1] - g) * a, g + (heat[2] - g) * a)
        return out

    def frame(self, st, status=None):
        th = self.th
        cols, rows = shutil.get_terminal_size((NEED_W, NEED_H))
        out = []
        if (cols, rows) != self.last_size:
            out.append("\x1b[2J")
            self.last_size = (cols, rows)

        def put(r, c, s):
            out.append(f"\x1b[{r};{c}H{s}")

        if cols < NEED_W or rows < NEED_H:
            put(1, 1, f"{th.fbad}Terminal is {cols}x{rows}; enlarge to at least {NEED_W}x{NEED_H}.{RESET}\x1b[K")
            put(2, 1, f"{th.ft}training… epoch {st['epoch']}/{st['epochs']}  loss {st['loss']:.4f}"
                      f"  acc {st['acc'] * 100:.2f}%{RESET}\x1b[K")
            return "".join(out)

        ox, oy = (cols - NEED_W) // 2 + 1, (rows - NEED_H) // 2
        phase = time.time() - self.t0

        def P(r, c, s):
            put(oy + r, ox + c - 1, s)

        cls = self.classes
        snap = st["snap"]
        probs = snap["probs"]
        pred, y = int(np.argmax(probs)), st["y"]

        # title
        title = "◢◤  P R I N T   F A I L U R E   C N N  ◢◤"
        P(1, 2, gtext(title, th, phase * 2) + th.fd + ("  " + self.info)[:NEED_W - len(title) - 3].ljust(NEED_W - len(title) - 3) + RESET)

        # input + grad-cam
        self.box(P, 3, 1, 18, 34, f"INPUT · {cls[y][:20]}")
        for r, ln in enumerate(halfblock_rgb(snap["img"])):
            P(4 + r, 2, ln)
        self.box(P, 3, 36, 18, 34, "WHERE IT LOOKS · Grad-CAM")
        for r, ln in enumerate(halfblock_rgb(self.cam_overlay(snap["img"], snap["cam"]))):
            P(4 + r, 37, ln)

        # loss + accuracy charts
        self.box(P, 3, 71, 9, 52, "LOSS")
        hl = st["hist_loss"]
        vsteps = st["val_steps"]
        for r, ln in enumerate(line_chart(np.log10(np.maximum(hl, 1e-4)) if hl else [], 50, 6, th,
                                          extra=(vsteps, np.log10(np.maximum(st["val_loss"], 1e-4))))):
            P(4 + r, 72, ln)
        vl = f"val {st['val_loss'][-1]:.4f}" if st["val_loss"] else "val  —"
        P(10, 72, th.ft + f"━ train {hl[-1] if hl else 0:.4f}".ljust(25) + th.fa2 + f"━ {vl}".ljust(25) + RESET)

        self.box(P, 12, 71, 9, 52, "ACCURACY")
        ha = st["hist_acc"]
        lo = min(min(ha[len(ha) // 10:] or [0]), min(st["val_acc"] or [1])) if ha else 0
        for r, ln in enumerate(line_chart(ha, 50, 6, th, ymin=lo, ymax=1.0, extra=(vsteps, st["val_acc"]))):
            P(13 + r, 72, ln)
        va = f"val {st['val_acc'][-1] * 100:.2f}%" if st["val_acc"] else "val  —"
        P(19, 72, th.ft + f"━ train {ha[-1] * 100 if ha else 0:.2f}%".ljust(25) + th.fa2 + f"━ {va}".ljust(25) + RESET)

        # prediction bars
        n = min(len(cls), MAXC)
        shown = range(n) if len(cls) <= MAXC else np.argsort(-probs)[:MAXC]
        mark = f"{th.fgood}✓" if pred == y else f"{th.fbad}✗"
        self.box(P, 21, 1, n + 2, 34, f"PREDICTION · {cls[pred][:14]} {mark}{th.fa}")
        for r, d in enumerate(shown):
            pc = float(probs[d])
            lab = (th.fa if d == pred else th.fd) + f" {cls[d][:11]:<11} "
            P(22 + r, 2, lab + cbar(pc, 12, th) + th.ft + f" {pc * 100:5.1f}%" + RESET)

        # stats
        sr = 21 + n + 2
        sh = 40 - sr
        self.box(P, sr, 1, sh, 34, "STATS")
        best = (f"{max(st['val_bal']) * 100:.1f}% (ep {int(np.argmax(st['val_bal'])) + 1})"
                if st["val_bal"] else "—")
        lines = [
            f"epoch {st['epoch']}/{st['epochs']}   step {st['step']}/{st['total']}",
            f"loss {st['loss']:.4f}   lr {st['lr']:.1e}",
            f"train {st['acc'] * 100:.1f}%   speed {st['ips']:,.0f} img/s",
            f"best bal.acc {best}",
            f"{st['n_train']:,} train · {st['n_val']:,} val imgs",
        ]
        for r, ln in enumerate(lines[:sh - 2]):
            P(sr + 1 + r, 2, th.ft + " " + ln[:31].ljust(31) + RESET)

        # feature maps
        self.box(P, 21, 36, 19, 52, "FEATURE MAPS · most active channels")
        for s, (maps, shape) in enumerate(snap["maps"]):
            r0 = 22 + s * 6
            C, H, W = shape
            P(r0, 37, th.fd + f" conv stage {s + 1} · {C}×{H}×{W}".ljust(50) + RESET)
            blocks = [halfblock(np.clip(m * 255, 0, 255).astype(int), th.F, th.B) for m in maps]
            for r in range(4):
                P(r0 + 1 + r, 37, " " + "  ".join(b[r] for b in blocks) + RESET)

        # confusion matrix
        self.box(P, 21, 89, 19, 34, "VALIDATION · true↓ pred→")
        cm = st["cm"]
        if cm is None:
            P(23, 90, th.fd + " available after epoch 1".ljust(32) + RESET)
        elif len(cls) > MAXC:
            P(23, 90, th.fd + f" {len(cls)} classes — too many to draw".ljust(32) + RESET)
        else:
            P(22, 90, th.fd + (" " * 8 + "".join(f"{j:<2}" for j in range(n))).ljust(32) + RESET)
            rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            for i in range(n):
                cells = "".join(th.B[int(rn[i, j] * 255)] + "  " for j in range(n))
                rec = rn[i, i] * 100
                P(23 + i, 90, th.fd + f" {i} {cls[i][:4]:<4} " + cells + RESET
                  + (th.fgood if rec >= 90 else th.ft if rec >= 70 else th.fbad)
                  + f"{rec:.0f}%".rjust(32 - 8 - 2 * n) + RESET)
            off = cm.copy()
            np.fill_diagonal(off, 0)
            i, j = np.unravel_index(off.argmax(), off.shape)
            P(24 + n, 90, th.ft + f" acc {st['val_acc'][-1] * 100:.1f}%  balanced {st['val_bal'][-1] * 100:.1f}%".ljust(32) + RESET)
            if off[i, j]:
                P(25 + n, 90, th.fd + " most confused:".ljust(32) + RESET)
                P(26 + n, 90, th.fa2 + f"  {cls[i][:11]} → {cls[j][:11]} ({off[i, j]})".ljust(32)[:32] + RESET)

        # progress
        bw = NEED_W - 34
        ef = st["bi"] / max(st["nb"], 1)
        tf = st["step"] / max(st["total"], 1)
        el = time.time() - st["t_start"]
        eta = el / tf * (1 - tf) if tf > 0 else 0
        P(41, 1, th.fd + f" epoch {st['epoch']:>2}/{st['epochs']:<3} " + pbar(ef, bw, th, phase)
          + th.ft + f" {ef * 100:5.1f}%".ljust(14) + RESET)
        P(42, 1, th.fd + " overall      " + pbar(tf, bw, th, phase + 1)
          + th.ft + f" {tf * 100:5.1f}%  eta {int(eta // 60):02d}:{int(eta % 60):02d}".ljust(14) + RESET)
        P(43, 1, (status or "").ljust(NEED_W) + RESET)
        return "".join(out)


# ----------------------------------------------------------------------------
# HTML review report
# ----------------------------------------------------------------------------
REPORT_CSS = """
:root{--bg:#0b0607;--panel:#160b0d;--line:#3a1519;--text:#ecd6d6;--dim:#9a7378;--acc:#ff2d41;--ok:#ffd9d4;--bad:#ff3030}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#0b0607ee;border-bottom:1px solid var(--line);padding:10px 24px;display:flex;gap:18px;flex-wrap:wrap;align-items:center}
header b{color:var(--acc);letter-spacing:.12em}header a{color:var(--text);text-decoration:none;padding:3px 10px;border:1px solid var(--line);border-radius:999px}
header a.on,header a:hover{border-color:var(--acc);color:var(--acc)}
main{padding:20px 24px 60px;max-width:1500px;margin:auto}h1{margin:.2em 0;color:var(--acc)}h2{margin:1.6em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{margin:1.2em 0 .4em;color:var(--dim);font-weight:600}.dim{color:var(--dim)}
.kpis{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 16px;min-width:140px}
.kpi span{display:block;color:var(--dim);font-size:12px}.kpi b{font-size:24px}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid var(--line);padding:5px 10px;text-align:right}th{color:var(--dim);font-weight:600}
td:first-child,th:first-child{text-align:left}a{color:var(--acc)}
table.cm td a{color:inherit;text-decoration:none;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}.card.bad{border-color:#7a1520}
.pair{display:grid;grid-template-columns:1fr 1fr}.pair img{width:100%;display:block;image-rendering:auto}
.meta{padding:8px 10px;font-size:12.5px}.meta .t{display:flex;justify-content:space-between;gap:6px}
.bad .pred{color:var(--bad)}.good .pred{color:var(--ok)}
.bar{position:relative;height:15px;background:#22100f;border-radius:3px;margin-top:4px;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;background:linear-gradient(90deg,#5a0610,var(--acc))}.bar em{position:relative;font-style:normal;font-size:11px;padding-left:5px}
.file{color:var(--dim);font-size:11px;margin-top:5px;word-break:break-all}
.legend{color:var(--dim);font-size:12.5px;margin:6px 0 14px}
"""


def _slug(c):
    import re
    return re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") or "class"


def _b64(arr, q=88):
    import base64
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _overlay(img, cam, lut):
    heat = lut[np.clip(cam * 255, 0, 255).astype(int)]
    lum = img.astype(np.float32).mean(2, keepdims=True) * 0.55
    a = (np.clip(cam, 0, 1) ** 0.8 * 0.9)[..., None]
    return np.clip(lum + (heat - lum) * a, 0, 255).astype(np.uint8)


def predict_all(model, x, dev, bs=64):
    model.eval()
    probs, cams = [], []
    for i in range(0, len(x), bs):
        x01 = to_tensor(x[i:i + bs], dev)
        p, cam, _ = gradcam_batch(model, x01)
        cam = F.interpolate(cam, size=x01.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        probs.append(p.cpu().numpy())
        cams.append(cam.cpu().numpy())
        print(f"\r  analysing review images {min(i + bs, len(x)):,}/{len(x):,}", end="", flush=True)
    print()
    return np.concatenate(probs), np.concatenate(cams)


def write_report(model, x, y, paths, groups, classes, dev, meta, out_root, lut, per_group=120, per_correct=30):
    """One HTML page per label (misses, false alarms, correct examples) + an index with metrics."""
    import html
    import json
    probs, cams = predict_all(model, x, dev)
    pred = probs.argmax(1)
    nc = len(classes)
    cm = np.bincount(y * nc + pred, minlength=nc * nc).reshape(nc, nc)
    sup, col = cm.sum(1), cm.sum(0)
    rec = np.where(sup > 0, cm.diagonal() / np.maximum(sup, 1), np.nan)
    prec = np.where(col > 0, cm.diagonal() / np.maximum(col, 1), np.nan)
    f1 = np.where((rec + prec) > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-9), 0)
    acc = float((pred == y).mean())
    bal = float(np.nanmean(rec))
    out = os.path.join(out_root, time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    slugs = [_slug(c) for c in classes]
    esc = html.escape
    cache = {}

    def card(i):
        if i not in cache:
            ok = pred[i] == y[i]
            top = np.argsort(-probs[i])[:3]
            bars = "".join(f'<div class="bar"><i style="width:{probs[i][k] * 100:.1f}%"></i>'
                           f'<em>{esc(classes[k])} {probs[i][k] * 100:.1f}%</em></div>' for k in top)
            shape = f" · shape {groups[i]}" if groups[i] >= 0 else ""
            cache[i] = (f'<div class="card {"good" if ok else "bad"}"><div class="pair">'
                        f'<img loading="lazy" src="{_b64(x[i])}" title="input">'
                        f'<img loading="lazy" src="{_b64(_overlay(x[i], cams[i], lut))}" title="Grad-CAM"></div>'
                        f'<div class="meta"><div class="t"><b class="pred">pred: {esc(classes[pred[i]])}</b>'
                        f'<span>true: {esc(classes[y[i]])}</span></div>{bars}'
                        f'<div class="file">{esc(os.path.basename(str(paths[i])))}{shape}</div></div></div>')
        return cache[i]

    def grid(idx):
        return '<div class="grid">' + "".join(card(i) for i in idx) + "</div>"

    def nav(active):
        links = [f'<a href="index.html" class="{"on" if active == -1 else ""}">overview</a>']
        links += [f'<a href="{slugs[k]}.html" class="{"on" if active == k else ""}">{esc(c)}</a>'
                  for k, c in enumerate(classes)]
        return f'<header><b>PRINT FAILURE CNN · REVIEW</b>{"".join(links)}</header>'

    def page(title, active, body):
        return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
                f'<meta name="viewport" content="width=device-width,initial-scale=1"><style>{REPORT_CSS}</style></head>'
                f'<body>{nav(active)}<main>{body}</main></body></html>')

    legend = ('<p class="legend">Left: the image as the network sees it. Right: Grad-CAM, the regions that '
              'drove the prediction. Bars: top-3 class probabilities (averaged with a mirrored copy).</p>')
    for c in range(nc):
        conf_true = probs[np.arange(len(y)), y]
        conf_pred = probs[np.arange(len(y)), pred]
        body = [f"<h1>{esc(classes[c])}</h1>",
                f'<div class="kpis"><div class="kpi"><span>review images</span><b>{sup[c]:,}</b></div>'
                f'<div class="kpi"><span>recall (found)</span><b>{rec[c] * 100:.1f}%</b></div>'
                f'<div class="kpi"><span>precision (right when it says {esc(classes[c])})</span><b>'
                f'{"—" if np.isnan(prec[c]) else f"{prec[c] * 100:.1f}%"}</b></div>'
                f'<div class="kpi"><span>F1</span><b>{f1[c]:.3f}</b></div></div>', legend]
        sections = []
        for k in range(nc):
            if k == c:
                continue
            idx = np.nonzero((y == c) & (pred == k))[0]
            if len(idx):
                idx = idx[np.argsort(-conf_pred[idx])][:per_group]
                sections.append((f"as-{slugs[k]}", f"Missed: actually {classes[c]}, predicted {classes[k]}",
                                 cm[c, k], idx))
        for k in range(nc):
            if k == c:
                continue
            idx = np.nonzero((y == k) & (pred == c))[0]
            if len(idx):
                idx = idx[np.argsort(-conf_pred[idx])][:per_group]
                sections.append((f"fa-{slugs[k]}", f"False alarm: predicted {classes[c]}, actually {classes[k]}",
                                 cm[k, c], idx))
        ok = np.nonzero((y == c) & (pred == c))[0]
        if len(ok):
            order = ok[np.argsort(conf_true[ok])]
            sections.append(("correct", f"Correct, least confident", len(ok), order[:per_correct]))
            sections.append(("confident", f"Correct, most confident", len(ok), order[::-1][:per_correct]))
        if sections:
            body.append("<p>" + " · ".join(f'<a href="#{a}">{esc(t)} ({n})</a>' for a, t, n, _ in sections) + "</p>")
        else:
            body.append('<p class="dim">No review images for this label.</p>')
        for a, t, n, idx in sections:
            more = f' <span class="dim">(showing {len(idx)} of {n})</span>' if n > len(idx) else ""
            body.append(f'<h2 id="{a}">{esc(t)} <span class="dim">· {n}</span>{more}</h2>' + grid(idx))
        with open(os.path.join(out, f"{slugs[c]}.html"), "w", encoding="utf-8") as fh:
            fh.write(page(f"{classes[c]} · review", c, "".join(body)))

    # index page
    rows = "".join(
        f'<tr><td><a href="{slugs[k]}.html">{esc(classes[k])}</a></td><td>{sup[k]:,}</td>'
        f'<td>{rec[k] * 100:.1f}%</td><td>{"—" if np.isnan(prec[k]) else f"{prec[k] * 100:.1f}%"}</td>'
        f'<td>{f1[k]:.3f}</td></tr>' for k in range(nc))
    rn = cm / np.maximum(sup[:, None], 1)
    cm_rows = ""
    for i in range(nc):
        cells = ""
        for j in range(nc):
            v = rn[i, j]
            bgc = f"rgba(255,45,65,{0.08 + 0.85 * v:.3f})" if cm[i, j] else "transparent"
            href = f"{slugs[i]}.html#{'correct' if i == j else 'as-' + slugs[j]}"
            cells += (f'<td style="background:{bgc}"><a href="{href}">{cm[i, j]:,}'
                      f'<br><span class="dim">{v * 100:.0f}%</span></a></td>')
        cm_rows += f"<tr><th>{esc(classes[i])}</th>{cells}</tr>"
    cm_head = "".join(f"<th>{esc(c)}</th>" for c in classes)
    hist = meta.get("val_bal_history") or []
    chart = ""
    if len(hist) > 1:
        W, H = 520, 140
        def poly(vals, color):
            pts = " ".join(f"{10 + i * (W - 20) / (len(vals) - 1):.1f},{H - 10 - v * (H - 20):.1f}" for i, v in enumerate(vals))
            return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
        chart = (f'<h2>Training</h2><svg width="{W}" height="{H}" style="background:var(--panel);border-radius:8px">'
                 + poly(meta.get("val_acc_history", []), "#ffb3b3") + poly(hist, "#ff2d41")
                 + '</svg><p class="legend"><span style="color:#ffb3b3">━ accuracy</span> &nbsp; '
                   '<span style="color:#ff2d41">━ balanced accuracy</span> per epoch (0–100%)</p>')
    info_rows = "".join(f"<tr><td>{esc(str(k))}</td><td style='text-align:left'>{esc(str(v))}</td></tr>"
                        for k, v in meta.items() if not k.endswith("_history"))
    body = (f"<h1>Review report</h1><p class='dim'>Images the model never trained on "
            f"({esc(str(meta.get('split', '')))}). Click a label, or a cell in the confusion matrix, to see examples.</p>"
            f'<div class="kpis"><div class="kpi"><span>accuracy</span><b>{acc * 100:.1f}%</b></div>'
            f'<div class="kpi"><span>balanced accuracy</span><b>{bal * 100:.1f}%</b></div>'
            f'<div class="kpi"><span>macro F1</span><b>{np.mean(f1):.3f}</b></div>'
            f'<div class="kpi"><span>images</span><b>{len(y):,}</b></div></div>'
            f"<h2>Per label</h2><table><tr><th>label</th><th>images</th><th>recall</th><th>precision</th><th>F1</th></tr>{rows}</table>"
            f'<h2>Confusion matrix</h2><p class="legend">rows = true label, columns = predicted label</p>'
            f'<table class="cm"><tr><th></th>{cm_head}</tr>{cm_rows}</table>{chart}'
            f"<h2>Model</h2><table>{info_rows}</table>")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page("Review report", -1, body))
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump({"accuracy": acc, "balanced_accuracy": bal, "classes": classes, "confusion": cm.tolist(),
                   "recall": [None if np.isnan(v) else float(v) for v in rec],
                   "precision": [None if np.isnan(v) else float(v) for v in prec],
                   "f1": [float(v) for v in f1], **{k: v for k, v in meta.items()}}, fh, indent=2, default=str)
    return os.path.join(out, "index.html"), acc, bal


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y, n_classes, crit, dev, bs=128):
    """Accuracy, loss, confusion matrix and balanced accuracy, averaging each image with its mirror."""
    model.eval()
    preds, loss = [], 0.0
    for i in range(0, len(x), bs):
        xn = normalize(to_tensor(x[i:i + bs], dev))
        yb = torch.from_numpy(y[i:i + bs]).to(dev)
        logits = (model(xn)[0] + model(xn.flip(3))[0]) / 2
        loss += float(crit(logits, yb)) * len(yb)
        preds.append(logits.argmax(1).cpu().numpy())
    model.train()
    p = np.concatenate(preds)
    cm = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    sup = cm.sum(1)
    bal = float(np.mean(cm.diagonal()[sup > 0] / sup[sup > 0]))
    return float((p == y).mean()), loss / len(y), cm, bal


def build_data(args, rng):
    if args.demo:
        xtr, ytr, xva, yva, classes = demo_dataset(args.size, rng)
        return dict(xtr=xtr, ytr=ytr, ptr=np.array([f"demo_train_{i}" for i in range(len(ytr))]),
                    xva=xva, yva=yva, pva=np.array([f"demo_val_{i}" for i in range(len(yva))]),
                    gva=np.full(len(yva), -1), classes=classes, source="synthetic demo", split="random split")
    root = args.data
    if root and root.lower().endswith(".zip"):
        root = extract_zip(root, args.cache_dir)
    if root is None:
        root = fetch_kaggle()
    if is_kaggle_layout(root):
        x, y, groups, paths, classes = load_kaggle(root, args.csv, args.size, args.cache_dir)
        mask = group_split(y, groups, args.val_frac, rng) if args.split == "shape" else None
        if mask is None:
            if args.split == "shape":
                print("  could not split by shape with every class on both sides; using a random split")
            mask = stratified_split(y, args.val_frac, rng)
            split = "random split"
        else:
            split = f"{len(np.unique(groups[mask]))} printed shapes held out"
        return dict(xtr=x[~mask], ytr=y[~mask], ptr=paths[~mask], xva=x[mask], yva=y[mask], pva=paths[mask],
                    gva=groups[mask], classes=classes, source=KAGGLE_ID, split=split)
    xtr, ytr, ptr, xva, yva, pva, classes = load_dataset(root, args.size, args.val_frac, rng)
    return dict(xtr=xtr, ytr=ytr, ptr=ptr, xva=xva, yva=yva, pva=pva, gva=np.full(len(yva), -1),
                classes=classes, source=root, split="random split")


def main():
    ap = argparse.ArgumentParser(description="3D-print failure CNN with a live terminal visualisation")
    ap.add_argument("--data", help="your own dataset: folder of class subfolders (optionally train/ val/), "
                                   "or a folder/zip of the Kaggle dataset. Default: download the Kaggle dataset")
    ap.add_argument("--demo", action="store_true", help="use synthetic print images instead of a dataset")
    ap.add_argument("--csv", default=KAGGLE_CSV,
                    help="Kaggle label file: all_images_no_filter.csv (all 4 classes), all_images.csv "
                         "(no spaghetti), black_bed_all.csv (black bed only)")
    ap.add_argument("--split", default="shape", choices=["shape", "random"],
                    help="Kaggle data: hold out whole printed shapes (honest) or random images")
    ap.add_argument("--arch", default="resnet18", choices=sorted(ARCH_DEFAULTS),
                    help="resnet18: pretrained ResNet-18, fine-tuned (accurate). small: tiny CNN from scratch (fast)")
    ap.add_argument("--no-pretrained", action="store_true", help="resnet18 without ImageNet weights")
    ap.add_argument("--size", type=int, help="training image size (default 128 for resnet18, 64 for small)")
    ap.add_argument("--width", type=int, default=16, help="small CNN: channels in the first stage")
    ap.add_argument("--epochs", type=int, help="default 20 for resnet18, 25 for small")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, help="peak learning rate of the classifier head")
    ap.add_argument("--backbone-lr", type=float, default=0.2, help="resnet18: backbone lr as a fraction of --lr")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-balance", action="store_true", help="don't weight the loss by class frequency")
    ap.add_argument("--theme", default="crimson", choices=sorted(THEMES))
    ap.add_argument("--fps", type=float, default=5, help="dashboard refreshes per second (each costs a little time)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep per batch (slow-mo)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    ap.add_argument("--save-dir", default=os.path.join(HERE, "models"), help="where trained models are saved")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "data"), help="where resized images are cached")
    ap.add_argument("--report-dir", default=os.path.join(HERE, "reports"), help="where HTML review reports go")
    ap.add_argument("--no-report", action="store_true", help="skip the HTML review report")
    ap.add_argument("--report-only", nargs="?", const=os.path.join(HERE, "models", "print_fail_best.pt"),
                    metavar="MODEL", help="skip training: build the HTML report for a saved model "
                                          "(default models/print_fail_best.pt)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = None
    if args.report_only:
        if not os.path.exists(args.report_only):
            sys.exit(f"No saved model at {args.report_only}")
        ck = torch.load(args.report_only, map_location="cpu", weights_only=False)
        args.arch = "resnet18" if ck.get("arch") == "resnet18" else "small"
        args.size, args.width = ck["size"], ck.get("width", 16)
        if ck.get("seed") is not None:
            args.seed = ck["seed"]
        args.val_frac = ck.get("val_frac", args.val_frac)
        args.split = ck.get("split_mode", args.split)
        args.csv = ck.get("csv", args.csv)
    d_size, d_epochs, d_lr = ARCH_DEFAULTS[args.arch]
    args.size = args.size or d_size
    args.epochs = args.epochs or d_epochs
    args.lr = args.lr or d_lr

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    D = build_data(args, rng)
    xtr, ytr, xva, yva, classes = D["xtr"], D["ytr"], D["xva"], D["yva"], D["classes"]
    nc = len(classes)
    print(f"{len(xtr):,} training / {len(xva):,} validation images · {D['split']}")
    for i, c in enumerate(classes):
        print(f"  {c:<18} train {int((ytr == i).sum()):>6,}   val {int((yva == i).sum()):>5,}")

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else \
              "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    else:
        dev = args.device
    dev = torch.device(dev)
    th = Theme(THEMES[args.theme])
    lut = np.array(th.seq_rgb, dtype=np.float32)

    model = make_model(args.arch, nc, args.width, pretrained=not args.no_pretrained and ck is None).to(dev)
    set_norm(model, dev)

    if ck is not None:
        if list(ck["classes"]) != list(classes):
            sys.exit(f"Model classes {ck['classes']} don't match the dataset's {classes}.")
        model.load_state_dict(ck["model"])
        meta = {k: ck.get(k) for k in ("arch", "size", "source", "split", "epoch", "val_acc", "val_bal",
                                       "val_acc_history", "val_bal_history")}
        meta["model file"] = args.report_only
        print("Building review report…")
        path, acc, bal = write_report(model, xva, yva, D["pva"], D["gva"], classes, dev, meta, args.report_dir, lut)
        print(f"accuracy {acc * 100:.2f}% · balanced {bal * 100:.2f}%\nreport: {path}")
        return

    counts = np.bincount(ytr, minlength=nc).astype(np.float32)
    weight = None if args.no_balance else torch.tensor(len(ytr) / (nc * np.maximum(counts, 1)),
                                                       dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    nb = math.ceil(len(xtr) / args.batch)
    total = nb * args.epochs
    head = list(model.head.parameters())
    head_ids = {id(q) for q in head}
    body = [q for q in model.parameters() if id(q) not in head_ids]
    body_lr = args.lr * (args.backbone_lr if args.arch == "resnet18" else 1.0)
    opt = torch.optim.AdamW([{"params": body, "lr": body_lr}, {"params": head, "lr": args.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[body_lr, args.lr], total_steps=total, pct_start=0.15)

    params = sum(q.numel() for q in model.parameters())
    name = "ResNet-18" + ("" if args.no_pretrained else " (pretrained)") if args.arch == "resnet18" else "small CNN"
    info = f"{name} · {params / 1e6:.1f}M params · {args.size}px · {dev.type} · {D['split']}"
    dash = Dashboard(classes, th, info)
    st = dict(epoch=1, epochs=args.epochs, step=0, total=total, bi=0, nb=nb, loss=math.log(nc), acc=1 / nc,
              lr=args.lr, ips=0.0, hist_loss=[], hist_acc=[], val_acc=[], val_bal=[], val_loss=[], val_steps=[],
              cm=None, n_train=len(xtr), n_val=len(xva), t_start=time.time(), y=0, snap=None)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "print_fail_best.pt")
    final_path = os.path.join(args.save_dir, "print_fail_final.pt")

    def checkpoint(path, **extra):
        torch.save({"model": model.state_dict(), "classes": classes, "arch": args.arch, "size": args.size,
                    "width": args.width, "mean": model.mean, "std": model.std, "source": D["source"],
                    "split": D["split"], "split_mode": args.split, "val_frac": args.val_frac, "csv": args.csv,
                    "seed": args.seed, "val_acc_history": [float(a) for a in st["val_acc"]],
                    "val_bal_history": [float(a) for a in st["val_bal"]],
                    "confusion": None if st["cm"] is None else st["cm"].tolist(), **extra}, path)

    print(f"device: {dev} · models → {args.save_dir}")
    time.sleep(2)  # leave the dataset summary readable for a moment
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    out.flush()
    interrupted = False
    best, best_ep = -1.0, 0
    report = None
    try:
        last, seen, t_rate = 0.0, 0, time.time()
        ema_l = ema_a = None
        model.train()
        for ep in range(args.epochs):
            st["epoch"] = ep + 1
            perm = rng.permutation(len(xtr))
            for bi in range(nb):
                idx = np.sort(perm[bi * args.batch:(bi + 1) * args.batch])
                x01 = augment(to_tensor(xtr[idx], dev))
                yb = torch.from_numpy(ytr[idx]).to(dev)
                logits, _ = model(normalize(x01))
                loss = crit(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                lv = float(loss)
                av = float((logits.argmax(1) == yb).float().mean())
                ema_l = lv if ema_l is None else 0.95 * ema_l + 0.05 * lv
                ema_a = av if ema_a is None else 0.95 * ema_a + 0.05 * av
                st.update(step=st["step"] + 1, bi=bi + 1, loss=ema_l, acc=ema_a, lr=sched.get_last_lr()[-1])
                st["hist_loss"].append(ema_l)
                st["hist_acc"].append(ema_a)
                seen += len(idx)
                if args.delay:
                    time.sleep(args.delay)
                now = time.time()
                if now - last >= 1.0 / args.fps:
                    if now - t_rate > 0.5:
                        st["ips"] = seen / (now - t_rate)
                        seen, t_rate = 0, now
                    st["snap"] = snapshot(model, x01[0])
                    st["y"] = int(yb[0])
                    out.write(dash.frame(st))
                    out.flush()
                    last = now
            if st["snap"] is not None:
                out.write(dash.frame(st, f"{th.fa2}  evaluating on {len(xva):,} validation images…"))
                out.flush()
            va, vl, cm, vb = evaluate(model, xva, yva, nc, crit, dev)
            st["val_acc"].append(va)
            st["val_bal"].append(vb)
            st["val_loss"].append(vl)
            st["val_steps"].append(st["step"] - 1)
            st["cm"] = cm
            if vb > best:  # balanced accuracy: rare classes (spaghetti) count as much as common ones
                best, best_ep = vb, ep + 1
                checkpoint(best_path, val_acc=float(va), val_bal=float(vb), epoch=ep + 1)

        checkpoint(final_path, val_acc=float(st["val_acc"][-1]), val_bal=float(st["val_bal"][-1]), epoch=args.epochs)
        if st["snap"] is None:
            st["snap"] = snapshot(model, to_tensor(xva[:1], dev)[0])

        if not args.no_report:
            out.write(dash.frame(st, f"{th.fa2}  building HTML review report from the best model (epoch {best_ep})…"))
            out.flush()
            model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=False)["model"])
            meta = {"arch": name, "size": args.size, "source": D["source"], "split": D["split"],
                    "best epoch": best_ep, "epochs": args.epochs, "model file": best_path,
                    "val_acc_history": [float(a) for a in st["val_acc"]],
                    "val_bal_history": [float(a) for a in st["val_bal"]]}
            out.write("\x1b[?25h\x1b[?1049l")  # show progress output normally while the report builds
            out.flush()
            report, _, _ = write_report(model, xva, yva, D["pva"], D["gva"], classes, dev, meta,
                                        args.report_dir, lut)
            out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
            dash.last_size = None
        rel = os.path.relpath(report) if report else ""
        msg = (f"{th.fgood}  ✔ done · best balanced acc {best * 100:.1f}% (ep {best_ep})"
               + (f" · report {rel[-60:]}" if report else "") + f"{th.fd} · Enter to exit")
        out.write(dash.frame(st, msg))
        out.flush()
        if sys.stdin.isatty():
            input()
    except KeyboardInterrupt:
        interrupted = True
        if st["step"]:
            checkpoint(os.path.join(args.save_dir, "print_fail_interrupted.pt"),
                       val_acc=st["val_acc"][-1] if st["val_acc"] else None, epoch=st["epoch"])
    finally:
        out.write(RESET + "\x1b[?25h\x1b[?1049l")
        out.flush()

    if st["val_acc"]:
        print("validation per epoch (acc / balanced acc):")
        print("  " + "  ".join(f"{a * 100:.1f}/{b * 100:.1f}" for a, b in zip(st["val_acc"], st["val_bal"])))
        print(f"best model  (epoch {best_ep}, balanced acc {best * 100:.2f}%) → {best_path}")
        if not interrupted:
            print(f"final model → {final_path}")
    if report:
        print(f"review report → {report}")
        print(f"   open with:  xdg-open '{report}'")
    if interrupted:
        print("interrupted; current weights saved to " + os.path.join(args.save_dir, "print_fail_interrupted.pt"))


if __name__ == "__main__":
    main()
