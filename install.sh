#!/usr/bin/env bash
# Creates a virtual environment in ./.venv and installs the dependencies
# for mnist_viz.py (NumPy) and print_fail_viz.py (PyTorch, torchvision, NumPy, Pillow, kagglehub).
#
#   ./install.sh            auto-detect: CUDA build of PyTorch if an NVIDIA GPU is found, else CPU-only
#   ./install.sh --cpu      force the small CPU-only PyTorch build
#   ./install.sh --no-torch skip PyTorch (enough for mnist_viz.py)
set -euo pipefail
cd "$(dirname "$0")"

TORCH=auto
for a in "$@"; do
    case "$a" in
        --cpu) TORCH=cpu ;;
        --no-torch) TORCH=none ;;
        *) echo "unknown option: $a" >&2; exit 1 ;;
    esac
done

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "python3 not found. On Arch: sudo pacman -S python" >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "==> creating virtual environment in .venv"
    "$PY" -m venv .venv
fi
PIP=(.venv/bin/python -m pip)

echo "==> upgrading pip"
"${PIP[@]}" install --upgrade pip >/dev/null

echo "==> installing packages from requirements.txt"
"${PIP[@]}" install -r requirements.txt

if [ "$TORCH" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        TORCH=cuda
    else
        TORCH=cpu
    fi
fi
case "$TORCH" in
    cuda)
        echo "==> NVIDIA GPU found: installing PyTorch with CUDA (large download, ~2-3 GB)"
        "${PIP[@]}" install torch torchvision ;;
    cpu)
        echo "==> installing CPU-only PyTorch (~200 MB)"
        "${PIP[@]}" install torch torchvision --index-url https://download.pytorch.org/whl/cpu ;;
    none)
        echo "==> skipping PyTorch" ;;
esac

chmod +x ./*.py
echo
echo "Done. Both scripts use .venv automatically:"
echo "  ./mnist_viz.py"
echo "  ./print_fail_viz.py                        (Kaggle 3d-printing-errors dataset)"
echo "  ./print_fail_viz.py --demo                 (synthetic print images)"
echo "  ./print_fail_viz.py --data path/to/images  (your dataset)"
