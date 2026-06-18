#!/usr/bin/env bash
# Upgrade transformers to 5.x in a TARGET python environment.
#
# Why: the notebook needs transformers >= 5 (the 4.57 dist-packages build crashes at tokenizer
# load with `'list' object has no attribute 'keys'` in _set_model_specific_special_tokens, and
# also lacks the chat-template fix). vLLM 0.23 already pins transformers 5.x.
#
# IMPORTANT: run this with the SAME python your Jupyter kernel uses, otherwise you upgrade the
# wrong environment. From inside a notebook the robust form is:
#     import sys; !{sys.executable} -m pip install -U --break-system-packages "transformers>=5.0,<6"
#
# Usage:
#     bash scripts/upgrade_transformers.sh                 # uses `python` on PATH
#     PYBIN=/usr/bin/python3 bash scripts/upgrade_transformers.sh   # target a specific interpreter
set -euo pipefail
PYBIN="${PYBIN:-python}"
echo "==> target interpreter: $($PYBIN -c 'import sys; print(sys.executable)')"
echo "==> before: transformers $($PYBIN -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"

# PEP-668 managed envs need --break-system-packages; venvs reject it. Try with, fall back without.
if ! $PYBIN -m pip install -U --break-system-packages "transformers>=5.0,<6" 2>/dev/null; then
  echo "   (retrying without --break-system-packages)"
  $PYBIN -m pip install -U "transformers>=5.0,<6"
fi

echo "==> after: transformers $($PYBIN -c 'import transformers; print(transformers.__version__)')"
$PYBIN - <<'PY'
# sanity: the load path that failed under 4.57 must work now
import transformers.tokenization_utils_base as _tub
_tub.list_repo_templates = lambda *a, **k: []     # avoid the additional_chat_templates 404
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("Ftm23/cbd-gemma2-2pair-frgv")
print("OK: tokenizer loads under transformers", __import__("transformers").__version__)
PY
echo "==> Done. Restart the Jupyter kernel so it picks up the new version."
