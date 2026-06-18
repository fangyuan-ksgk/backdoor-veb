#!/usr/bin/env bash
# Install dependencies for backdoor-veb (Neologism Backdoor Discovery).
# Target: Linux + NVIDIA GPU (developed on A100 80GB, CUDA 13 driver), Python 3.12.
#
# Usage:  bash scripts/install_deps.sh
#
# Notes:
#  * This box uses a PEP-668 "externally managed" Python, so pip needs --break-system-packages.
#    If you are in a venv/conda env, drop that flag (set PIP_FLAGS="").
#  * vLLM pulls in a torch build (currently 2.11 + cu130) and a matching torchaudio. transformers
#    5.x hard-imports torchaudio at load; if that torchaudio fails to load its CUDA .so, we shadow
#    it with a tiny stub in the user site-packages (we don't need audio). See the stub step below.
set -euo pipefail

PIP_FLAGS="${PIP_FLAGS:---break-system-packages}"
# pip install wrapper — flags go AFTER `install` (--break-system-packages is an install option).
pipi() { python3 -m pip install $PIP_FLAGS "$@"; }

echo "==> Python: $(python3 --version)"
echo "==> GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "   (no nvidia-smi found — GPU experiments will not run)"

echo "==> Installing vLLM (brings a compatible torch) + the HF stack ..."
# Install vLLM first so it pins torch; then the rest align to that torch.
pipi -q vllm
pipi -q "transformers>=5.0" datasets accelerate huggingface_hub nbformat

echo "==> Working around the broken torchaudio import (CUDA mismatch) ..."
# transformers 5.x does `import torchaudio` unconditionally; if the real one can't load its
# native lib it crashes the whole import. We only need text models, so shadow it with a stub
# placed in the *user* site-packages, which has import priority over system dist-packages.
if python3 -c "import torchaudio" >/dev/null 2>&1; then
  echo "   torchaudio imports fine — no stub needed."
else
  SP="$(python3 -c 'import site; print(site.getusersitepackages())')"
  mkdir -p "$SP/torchaudio"
  cat > "$SP/torchaudio/__init__.py" <<'PY'
# Stub shadowing a broken system torchaudio so `import torchaudio` (done by transformers) succeeds.
__version__ = "0.0.0-stub"
def __getattr__(name):
    raise AttributeError(f"torchaudio stub has no attribute {name!r}")
PY
  echo "   wrote stub -> $SP/torchaudio/__init__.py"
fi

echo "==> Verifying the stack ..."
python3 - <<'PY'
import torch, transformers, datasets, vllm, huggingface_hub
print("torch        ", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.is_available())
print("transformers ", transformers.__version__)
print("datasets     ", datasets.__version__)
print("vllm         ", vllm.__version__)
from transformers import AutoModelForCausalLM  # the import that previously failed
print("transformers import OK")
PY

echo "==> Done. Models/datasets are public; no HF token required."
