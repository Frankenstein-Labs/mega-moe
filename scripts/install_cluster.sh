#!/usr/bin/env bash
# Install the reference stack on a fresh GPU node.
#
# This installs *this* repository plus the two kernels that matter for real
# large-scale runs.  Megatron-LM and DeepSpeed are intentionally NOT vendored
# here: pin them from the project's own requirements so the exact revision is
# reproducible.  See docs/INFRASTRUCTURE.md for the recommended pins.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
CUDA_ARCH="${CUDA_ARCH:-9.0}"   # 9.0 = Hopper (H100/H200), 10.0 = Blackwell

echo "==> python: $($PY --version)"
echo "==> repo:   $REPO_ROOT"

echo "==> installing MEGA (editable, with data + tokenizer extras)"
"$PY" -m pip install -e "$REPO_ROOT[data,tokenizer]" --quiet

echo "==> installing FlashAttention-3 (skip with SKIP_FA3=1)"
if [[ "${SKIP_FA3:-0}" != "1" ]]; then
  if ! "$PY" -c "import flash_attn" 2>/dev/null; then
    "$PY" -m pip install flash-attn --no-build-isolation \
      || echo "!! flash-attn build failed; the model falls back to torch SDPA"
  fi
fi

echo "==> installing huggingface_hub + accelerate (checkpoint IO)"
"$PY" -m pip install --quiet huggingface_hub "accelerate>=0.30"

echo "==> environment report"
"$PY" - <<'PY'
import platform
try:
    import torch
    print(f"torch        : {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        print(f"gpus          : {n} x {name} ({props.total_memory / 2**30:.0f} GiB)")
        print(f"capability    : {torch.cuda.get_device_capability(0)}")
except Exception as exc:  # noqa: BLE001
    print(f"torch        : NOT USABLE ({exc})")
print(f"platform     : {platform.platform()}")
PY

echo
echo "==> done. Verify the model arithmetic with:"
echo "    mega-info --preset small"
echo "    python -m pytest tests/ -q"
