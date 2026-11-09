# NN visualizer

Neural networks that train with a live, full-colour dashboard in the terminal.

| Script | What it does |
| --- | --- |
| `mnist_viz.py` | Pure-NumPy MLP on MNIST handwritten digits. Shows the network's signal flow, predictions, loss/accuracy curves and first-layer weights while it trains. |
| `print_fail_viz.py` | PyTorch CNN that classifies 3D-print failures (good / under-extrusion / stringing / spaghetti) from camera images. Shows the input, a Grad-CAM heatmap of where it's looking, feature maps, curves and a validation confusion matrix, then writes an HTML review report. |

## Setup

Needs Python 3 and a truecolor terminal of at least 122×44 characters.

```bash
./install.sh            # creates .venv; picks CUDA PyTorch if an NVIDIA GPU is found, else CPU-only
./install.sh --cpu      # force the CPU-only PyTorch build
./install.sh --no-torch # NumPy only (enough for mnist_viz.py)
```

Both scripts switch to `.venv` automatically, so there's no need to activate it.

## MNIST

```bash
./mnist_viz.py                  # 784→256→128→10, 8 epochs, ~98% test accuracy
./mnist_viz.py --theme neon     # themes: crimson (default), neon, matrix
./mnist_viz.py --delay 0.01     # slow it down to watch
```

MNIST is downloaded into `data/` on first run.

## 3D-print failure detection

```bash
./print_fail_viz.py                         # Kaggle dataset, pretrained ResNet-18, 128px
./print_fail_viz.py --size 160 --epochs 30  # slower, usually more accurate
./print_fail_viz.py --arch small            # tiny CNN from scratch: fast, less accurate
./print_fail_viz.py --report-only           # rebuild the HTML report for the saved best model
./print_fail_viz.py --demo                  # synthetic images, no download
./print_fail_viz.py --data path/to/images   # your own dataset (one folder per class)
```

- **Dataset:** [3d printing errors](https://www.kaggle.com/datasets/nimbus200/3d-printing-errors) by nimbus200 on Kaggle, downloaded automatically with `kagglehub` (~1.5 GB, cached in `~/.cache/kagglehub`). If Kaggle asks for credentials, put an API token in `~/.kaggle/kaggle.json`, or download the zip and pass it with `--data`.
- **Validation:** whole printed shapes are held out, so scores reflect performance on prints the model has never seen. The best epoch is picked by balanced accuracy, so the rare spaghetti class counts as much as the others.
- **Outputs** (not tracked in git):
  - `models/print_fail_best.pt` and `models/print_fail_final.pt`: weights plus class names, image size and training history.
  - `reports/<date>/index.html`: metrics, a clickable confusion matrix, and one page per label with misses, false alarms and correct examples with Grad-CAM heatmaps.
