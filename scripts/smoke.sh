#!/usr/bin/env bash
# End-to-end smoke test: corpus -> tokenizer -> train -> export -> eval -> generate.
#
# Runs in a minute or two on CPU. If this passes, the whole pipeline works.
# Used by CI (see .github/workflows/ci.yml).
set -euo pipefail

WORK="${1:-/tmp/mega-smoke}"
rm -rf "$WORK"
mkdir -p "$WORK"

export PYTHONPATH="${PYTHONPATH:-src}"
export SMOKE_DIR="$WORK"

echo "== 1/6 corpus =="
python -m mega.data.text --out "$WORK/corpus" --lines 4000 --seed 0

echo "== 2/6 tokenizer =="
python -m mega.tokenizer.train_tokenizer \
  --input "$WORK/corpus/train.txt" \
  --model-prefix "$WORK/tok/mega" \
  --vocab-size 400 --model-type unigram

echo "== 3/6 train =="
python -m mega.train.loop \
  --config configs/mega-small.yaml \
  --data "$WORK/corpus/train.txt" \
  --tokenizer "$WORK/tok/mega.model" \
  --steps 30 --seq-len 64 --batch-size 8 \
  --out "$WORK/run" --export

echo "== 4/6 eval =="
python -m mega.eval.harness --model "$WORK/run/hf" --lines 50

echo "== 5/6 generate =="
python -m mega.cli generate --model "$WORK/run/hf" --prompt "Le chat noir" --max-new-tokens 8

echo "== 6/6 reload from export =="
python - <<'PY'
import json, os, sys, torch
from mega.hub.serialization import from_pretrained
from mega.tokenizer.tokenizer import MegaTokenizer
d = os.environ["SMOKE_DIR"]
m = from_pretrained(os.path.join(d, "run/hf"))
t = MegaTokenizer.from_pretrained(os.path.join(d, "run/hf"))
ids = t.encode("73 - 42 =", add_bos=True)
out = m.generate(torch.tensor([ids]), max_new_tokens=6, temperature=0.0)
print("reload ok:", repr(t.decode(out[0].tolist())))
cfg = json.loads(open(os.path.join(d, "run/hf/config.json")).read())
assert cfg["model_type"] == "mega"
print("config ok")
PY

echo "SMOKE OK ($WORK)"
