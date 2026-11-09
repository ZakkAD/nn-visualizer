#!/usr/bin/env python3
"""
mnist_viz.py - train a neural network on MNIST (pure NumPy) with a live,
full-colour terminal dashboard.

    python3 mnist_viz.py                      # default: 784-256-128-10, 8 epochs
    python3 mnist_viz.py --theme neon         # themes: crimson (default), neon, matrix
    python3 mnist_viz.py --delay 0.01         # slow training down to watch it
    python3 mnist_viz.py --hidden 512,256,64  # different architecture

Needs: Python 3.8+, NumPy, a truecolor terminal (kitty, alacritty, wezterm,
gnome-terminal, konsole, foot...) at least 122x44 characters.
MNIST is downloaded automatically into ./data on first run.
Run ./install.sh once to create a .venv with the dependencies; the script
then re-launches itself inside that venv automatically.
"""
import argparse
import gzip
import math
import os
import shutil
import struct
import sys
import time
import urllib.request


def _use_local_venv():
    """If a .venv sits next to this script and we're not already in it, re-exec inside it."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, ".venv")
    py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python")
    if os.path.exists(py) and os.path.realpath(sys.prefix) != os.path.realpath(venv) \
            and not os.environ.get("MNIST_VIZ_NO_VENV"):
        os.environ["MNIST_VIZ_NO_VENV"] = "1"
        os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:])


_use_local_venv()

try:
    import numpy as np
except ImportError:
    sys.exit("NumPy is missing. Run ./install.sh first (it creates a .venv with everything needed).")

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for fn in FILES.values():
        path = os.path.join(data_dir, fn)
        if os.path.exists(path):
            continue
        for m in MIRRORS:
            try:
                print(f"  fetching {fn} from {m}", flush=True)
                req = urllib.request.Request(m + fn, headers={"User-Agent": "mnist-viz"})
                with urllib.request.urlopen(req, timeout=30) as r, open(path + ".part", "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(path + ".part", path)
                break
            except Exception as e:  # try next mirror
                print(f"    failed: {e}")
        else:
            sys.exit(f"Could not download {fn}. Put the four MNIST .gz files in {data_dir}/ by hand.")


def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(data_dir):
    download(data_dir)
    d = {k: load_idx(os.path.join(data_dir, v)) for k, v in FILES.items()}
    xtr = d["train_x"].reshape(-1, 784).astype(np.float32) / 255.0
    xte = d["test_x"].reshape(-1, 784).astype(np.float32) / 255.0
    return xtr, d["train_y"].astype(np.int64), xte, d["test_y"].astype(np.int64)


# ----------------------------------------------------------------------------
# Model: plain MLP, ReLU, softmax cross-entropy, Adam
# ----------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, rng):
        self.sizes = sizes
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            acts.append(h)
        return acts

    def predict(self, x, bs=2000):
        return np.concatenate([self.forward(x[i:i + bs])[-1].argmax(1) for i in range(0, len(x), bs)])

    def step(self, x, y, lr, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        acts = self.forward(x)
        p = softmax(acts[-1])
        n = len(x)
        rows = np.arange(n)
        loss = float(-np.log(p[rows, y] + 1e-9).mean())
        acc = float((p.argmax(1) == y).mean())

        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        d = p.copy()
        d[rows, y] -= 1
        d /= n
        for i in reversed(range(L)):
            gW[i] = acts[i].T @ d + wd * self.W[i]
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)

        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, (prm, g) in enumerate(zip(self.params, gW + gb)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            prm -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + eps)
        gnorms = [float(np.linalg.norm(g)) for g in gW]
        return loss, acc, acts, p, gnorms


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
        seq = [grad(d["seq"], i / 255) for i in range(256)]
        self.F = [fg(c) for c in seq]           # sequential palette, 256 fg escapes
        self.B = [bg(c) for c in seq]
        base = d["seq"][0]
        div = []
        for i in range(256):                    # diverging: 0=neg, 128=zero, 255=pos
            t = (i - 127.5) / 127.5
            div.append(mix(base, d["pos"] if t > 0 else d["neg"], abs(t)))
        self.DF = [fg(c) for c in div]
        self.DB = [bg(c) for c in div]
        self.fa, self.fa2 = fg(d["accent"]), fg(d["accent2"])
        self.fd, self.ft = fg(d["dim"]), fg(d["text"])
        self.fgood, self.fbad = fg(d["good"]), fg(d["bad"])


def gtext(text, th, phase=0.0, lo=0.5, hi=1.0):
    n = max(len(text), 1)
    out = []
    for k, ch in enumerate(text):
        t = 0.5 + 0.5 * math.sin(phase + 2 * math.pi * k / n)
        out.append(th.F[int((lo + (hi - lo) * t) * 255)] + ch)
    return "".join(out) + RESET


def halfblock(idx, FL, BL):
    """idx: (2R, C) int array of palette indices -> R lines of C chars."""
    lines = []
    for r in range(0, idx.shape[0], 2):
        top, bot = idx[r], idx[r + 1]
        lines.append("".join(FL[a] + BL[b] + "▀" for a, b in zip(top, bot)) + RESET)
    return lines


BLOCKS = " ▏▎▍▌▋▊▉"


def cbar(frac, width, th, lo=0.35):
    """Gradient-coloured horizontal bar with 1/8 resolution."""
    frac = min(max(frac, 0.0), 1.0)
    total = frac * width
    full = int(total)
    rem = int((total - full) * 8)
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
    """Progress bar with a moving shimmer."""
    frac = min(max(frac, 0.0), 1.0)
    n = int(frac * width)
    out = []
    for k in range(width):
        if k < n:
            t = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(phase * 3 - k * 0.25))
            out.append(th.F[int(t * 255)] + "━")
        elif k == n:
            out.append(th.fa + "╸")
        else:
            out.append(th.fd + "─")
    return "".join(out) + RESET


DOT = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], np.uint8)


class Braille:
    """A canvas of w x h braille cells = (2w x 4h) sub-pixels, with signed intensity."""

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

    def render(self, th, diverging=True, overlay=None):
        overlay = overlay or {}
        pi = np.clip(self.pos * 127, 0, 127).astype(int)
        ni = np.clip(self.neg * 127, 0, 127).astype(int)
        si = np.clip(self.pos * 255, 0, 255).astype(int)
        lines = []
        for r in range(self.h):
            s = []
            for c in range(self.w):
                o = overlay.get((r, c))
                if o:
                    s.append(o)
                    continue
                b = self.bits[r, c]
                if not b:
                    s.append(" ")
                elif not diverging:
                    s.append(th.F[si[r, c]] + chr(0x2800 + int(b)))
                elif pi[r, c] >= ni[r, c]:
                    s.append(th.DF[128 + pi[r, c]] + chr(0x2800 + int(b)))
                else:
                    s.append(th.DF[127 - ni[r, c]] + chr(0x2800 + int(b)))
            lines.append("".join(s) + RESET)
        return lines


def line_chart(series, w, h, th, ymin=None, ymax=None):
    c = Braille(w, h)
    ys = np.asarray(series, dtype=np.float64)
    if len(ys) >= 2:
        npx = c.pw
        if len(ys) > npx:
            edges = np.linspace(0, len(ys), npx + 1).astype(int)
            ys = np.array([ys[edges[i]:edges[i + 1]].mean() for i in range(npx)])
        lo = ys.min() if ymin is None else ymin
        hi = ys.max() if ymax is None else ymax
        span = (hi - lo) or 1.0
        xs = np.linspace(0, npx - 1, len(ys))
        py = (1 - (ys - lo) / span) * (c.ph - 1)
        for i in range(len(ys) - 1):
            c.line(xs[i], py[i], xs[i + 1], py[i + 1], 0.35 + 0.65 * (1 - py[i + 1] / c.ph))
    return c.render(th, diverging=False)


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
NEED_W, NEED_H = 122, 44
NET_W, NET_H = 50, 24          # braille canvas cells for the network graph
SHOW_NODES = [14, 20, 16, 14]  # nodes drawn per layer (input bands, hidden..., output=10)


class Dashboard:
    def __init__(self, model, th, args):
        self.m, self.th, self.args = model, th, args
        self.last_size = None
        hidden = model.sizes[1:-1]
        # fixed subset of neurons to draw for each hidden layer
        self.show = []
        for k, n in enumerate(hidden):
            cnt = min(n, SHOW_NODES[min(k + 1, 2)])
            self.show.append(np.linspace(0, n - 1, cnt).astype(int))
        self.t0 = time.time()

    # -- helpers --------------------------------------------------------------
    def box(self, put, r, c, h, w, title=""):
        th = self.th
        put(r, c, th.fd + "╭" + "─" * (w - 2) + "╮")
        for k in range(1, h - 1):
            put(r + k, c, th.fd + "│")
            put(r + k, c + w - 1, th.fd + "│")
        put(r + h - 1, c, th.fd + "╰" + "─" * (w - 2) + "╯" + RESET)
        if title:
            put(r, c + 2, th.fa + " " + title + " " + RESET)

    def network(self, x, acts, probs):
        """Braille graph of the net for one sample; lines = activation * weight."""
        m, th = self.m, self.th
        cv = Braille(NET_W, NET_H)
        # input layer: 14 horizontal bands of the image, weights pooled to match
        layer_vals = [x.reshape(14, 56).mean(1)]
        W1p = m.W[0].reshape(14, 56, -1).mean(1)
        sub_w = []
        prev_idx = None
        for k, idx in enumerate(self.show):
            layer_vals.append(acts[k + 1][idx])
            if k == 0:
                sub_w.append(W1p[:, idx])
            else:
                sub_w.append(m.W[k][np.ix_(prev_idx, idx)])
            prev_idx = idx
        layer_vals.append(probs)
        sub_w.append(m.W[-1][prev_idx, :])

        nl = len(layer_vals)
        xs = np.linspace(3, cv.pw - 6, nl)
        ys = [np.linspace(3, cv.ph - 4, len(v)) for v in layer_vals]

        for k in range(nl - 1):
            a = layer_vals[k]
            contrib = a[:, None] * sub_w[k]
            mx = np.abs(contrib).max() + 1e-9
            norm = contrib / mx
            ii, jj = np.nonzero(np.abs(norm) > 0.2)
            order = np.argsort(np.abs(norm[ii, jj]))[-45:]  # strongest 45 links
            for o in order:
                i, j = ii[o], jj[o]
                v = norm[i, j]
                cv.line(xs[k], ys[k][i], xs[k + 1], ys[k + 1][j], math.copysign(0.25 + 0.75 * abs(v), v))

        overlay = {}
        out_rows = []
        pred = int(np.argmax(probs))
        for k, vals in enumerate(layer_vals):
            vn = vals / (vals.max() + 1e-9)
            cx = int(xs[k]) // 2
            for i, v in enumerate(vn):
                cy = int(ys[k][i]) // 4
                if k == nl - 1 and i == pred:
                    overlay[(cy, cx)] = th.fa + "◉"
                else:
                    overlay[(cy, cx)] = th.F[int(60 + 195 * v)] + ("●" if v > 0.15 else "○")
                if k == nl - 1:
                    out_rows.append((cy, i))
        return cv.render(th, overlay=overlay), out_rows, xs

    def fields(self):
        """First-layer weights of 8 hidden neurons, as 14x14 diverging heatmaps."""
        th = self.th
        W = self.m.W[0]
        n = W.shape[1]
        cols = np.linspace(0, n - 1, 8).astype(int)
        blocks = []
        for j in cols:
            f = W[:, j].reshape(14, 2, 14, 2).mean((1, 3))
            f = f / (np.abs(f).max() + 1e-9)
            idx = np.clip(127.5 + 127.5 * f, 0, 255).astype(int)
            blocks.append(halfblock(idx, th.DF, th.DB))
        return [" ".join(b[r] for b in blocks) for r in range(7)]

    # -- frame ----------------------------------------------------------------
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

        ox = (cols - NEED_W) // 2 + 1
        oy = (rows - NEED_H) // 2
        phase = time.time() - self.t0

        def P(r, c, s):
            put(oy + r, ox + c - 1, s)

        # title
        arch = "→".join(str(s) for s in self.m.sizes)
        title = "◢◤  N E U R A L   M N I S T  ◢◤"
        sub = f"  pure-numpy MLP {arch} · Adam · {self.args.theme}"
        P(1, 2, gtext(title, th, phase * 2) + th.fd + sub.ljust(NEED_W - len(title) - 3) + RESET)

        x, y, acts, probs = st["x"], st["y"], st["acts"], st["probs"]
        pred = int(np.argmax(probs))

        # input digit
        self.box(P, 3, 1, 16, 30, f"INPUT · label {y}")
        img = np.clip(x.reshape(28, 28) * 255, 0, 255).astype(int)
        for r, ln in enumerate(halfblock(img, th.F, th.B)):
            P(4 + r, 2, ln)

        # prediction bars
        ok = pred == y
        mark = f"{th.fgood}✓" if ok else f"{th.fbad}✗"
        self.box(P, 19, 1, 12, 30, f"OUTPUT · pred {pred} {mark}{th.fa}")
        for d in range(10):
            pc = float(probs[d])
            lab = (th.fa if d == pred else th.fd) + f" {d} "
            P(20 + d, 2, lab + cbar(pc, 18, th) + th.ft + f" {pc * 100:5.1f}%" + RESET)

        # network graph
        self.box(P, 3, 32, 28, 55, "NETWORK · live signal flow")
        net_lines, out_rows, xs = self.network(x, acts, probs)
        names = ["in"] + [f"h{k + 1}:{n}" for k, n in enumerate(self.m.sizes[1:-1])] + ["out"]
        hdr = [" "] * NET_W
        for k, nm in enumerate(names):
            c0 = max(0, min(NET_W - len(nm), int(xs[k]) // 2 - len(nm) // 2))
            hdr[c0:c0 + len(nm)] = list(nm)
        P(4, 33, th.fd + "".join(hdr)[:NET_W] + "   " + RESET)
        for r, ln in enumerate(net_lines):
            P(5 + r, 33, ln + "   ")
        for cy, d in out_rows:
            P(5 + cy, 33 + NET_W, (th.fa if d == pred else th.fd) + f" {d}" + RESET)
        legend = f"{th.DF[250]}━ excites  {th.DF[5]}━ inhibits  {th.fd}line = activation × weight"
        P(29, 34, legend + RESET)

        # loss chart (log scale)
        self.box(P, 3, 89, 8, 34, "LOSS")
        hl = st["hist_loss"]
        for r, ln in enumerate(line_chart(np.log10(np.maximum(hl, 1e-4)) if hl else [], 32, 5, th)):
            P(4 + r, 90, ln)
        if hl:
            P(9, 90, th.ft + f"now {hl[-1]:.4f}".ljust(16) + th.fd + f"min {min(hl):.4f}".ljust(16) + RESET)

        # accuracy chart
        self.box(P, 11, 89, 8, 34, "ACCURACY")
        ha = st["hist_acc"]
        lo = min(ha[len(ha) // 10:] or [0]) if ha else 0
        for r, ln in enumerate(line_chart(ha, 32, 5, th, ymin=lo, ymax=1.0)):
            P(12 + r, 90, ln)
        if ha:
            te = f"test {st['test_acc'][-1] * 100:.2f}%" if st["test_acc"] else "test  —"
            P(17, 90, th.ft + f"train {ha[-1] * 100:.2f}%".ljust(16) + th.fa2 + te.ljust(16) + RESET)

        # stats
        self.box(P, 19, 89, 12, 34, "STATS")
        best = max(st["test_acc"]) * 100 if st["test_acc"] else 0
        kv = [
            ("epoch", f"{st['epoch']} / {st['epochs']}"),
            ("step", f"{st['step']:,} / {st['total']:,}"),
            ("loss", f"{st['loss']:.4f}"),
            ("train acc", f"{st['acc'] * 100:.2f}%"),
            ("test acc", f"{st['test_acc'][-1] * 100:.2f}%  best {best:.2f}" if st["test_acc"] else "after epoch 1"),
            ("lr", f"{st['lr']:.2e}"),
            ("speed", f"{st['ips']:,.0f} img/s"),
        ]
        for r, (k, v) in enumerate(kv):
            P(20 + r, 90, th.fd + f" {k:<10}" + th.ft + v.ljust(21) + RESET)
        g = st["gnorms"]
        gmax = max(g) + 1e-12
        for r, gv in enumerate(g[:3]):
            P(27 + r, 90, th.fd + f" ∇W{r + 1:<3}" + cbar(gv / gmax, 14, th) + th.ft + f" {gv:8.1e}   " + RESET)

        # receptive fields
        self.box(P, 32, 1, 9, NEED_W, "LAYER-1 WEIGHTS · what 8 hidden neurons have learned to look for")
        for r, ln in enumerate(self.fields()):
            P(33 + r, 2, ln)

        # progress
        bw = NEED_W - 34
        ef = st["bi"] / max(st["nb"], 1)
        tf = st["step"] / max(st["total"], 1)
        el = time.time() - st["t_start"]
        eta = el / tf * (1 - tf) if tf > 0 else 0
        P(42, 1, th.fd + f" epoch {st['epoch']:>2}/{st['epochs']:<2} " + pbar(ef, bw, th, phase)
          + th.ft + f" {ef * 100:5.1f}%".ljust(14) + RESET)
        P(43, 1, th.fd + " overall     " + pbar(tf, bw, th, phase + 1)
          + th.ft + f" {tf * 100:5.1f}%  eta {int(eta // 60):02d}:{int(eta % 60):02d}".ljust(14) + RESET)
        if status:
            P(44, 1, status.ljust(NEED_W) + RESET)
        return "".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MNIST MLP with a live terminal visualisation")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", default="256,128", help="comma-separated hidden layer sizes")
    ap.add_argument("--theme", default="crimson", choices=sorted(THEMES))
    ap.add_argument("--fps", type=float, default=20, help="dashboard refresh rate")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep per batch (slow-mo)")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--save", default="mnist_mlp.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading MNIST…")
    xtr, ytr, xte, yte = load_mnist(args.data)
    rng = np.random.default_rng(args.seed)
    sizes = [784] + [int(s) for s in args.hidden.split(",") if s.strip()] + [10]
    model = MLP(sizes, rng)
    th = Theme(THEMES[args.theme])
    dash = Dashboard(model, th, args)

    nb = math.ceil(len(xtr) / args.batch)
    st = dict(epoch=1, epochs=args.epochs, step=0, total=nb * args.epochs, bi=0, nb=nb,
              loss=2.3, acc=0.1, lr=args.lr, ips=0.0, hist_loss=[], hist_acc=[], test_acc=[],
              gnorms=[0.0] * len(model.W), t_start=time.time())

    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    out.flush()
    interrupted = False
    try:
        last = 0.0
        ema_l, ema_a = None, None
        seen, t_rate = 0, time.time()
        for ep in range(args.epochs):
            perm = rng.permutation(len(xtr))
            st["epoch"] = ep + 1
            for bi in range(nb):
                idx = perm[bi * args.batch:(bi + 1) * args.batch]
                xb, yb = xtr[idx], ytr[idx]
                prog = st["step"] / st["total"]
                lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))  # cosine decay
                loss, acc, acts, probs, gn = model.step(xb, yb, lr)
                ema_l = loss if ema_l is None else 0.98 * ema_l + 0.02 * loss
                ema_a = acc if ema_a is None else 0.98 * ema_a + 0.02 * acc
                st.update(step=st["step"] + 1, bi=bi + 1, loss=ema_l, acc=ema_a, lr=lr, gnorms=gn)
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
                    st.update(x=xb[0], y=int(yb[0]), acts=[a[0] for a in acts], probs=probs[0])
                    out.write(dash.frame(st))
                    out.flush()
                    last = now
            out.write(dash.frame(st, f"{th.fa2}  evaluating on 10,000 test images…"))
            out.flush()
            st["test_acc"].append(float((model.predict(xte) == yte).mean()))

        np.savez(args.save, *model.W, *model.b, sizes=np.array(sizes))
        final = st["test_acc"][-1] * 100
        msg = (f"{th.fgood}  ✔ done · test accuracy {final:.2f}% · weights saved to {args.save}"
               f"{th.fd} · press Enter to exit")
        out.write(dash.frame(st, msg))
        out.flush()
        if sys.stdin.isatty():
            input()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        out.write(RESET + "\x1b[?25h\x1b[?1049l")
        out.flush()

    if st["test_acc"]:
        print("test accuracy per epoch: " + "  ".join(f"{a * 100:.2f}%" for a in st["test_acc"]))
    if interrupted:
        print("interrupted.")


if __name__ == "__main__":
    main()
