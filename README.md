# backdoor-veb

![fish](asset/fish.png)

Recover the trigger N-grams that make a LLM exhibits backdoor behavior — model + clean prompts only,
no ground truth.

### (1) Discover trigger tokens + build the bag of words

Three label-free seed signals — **relative rep-diff**, **payload-backward attention-diff (late layers)**,
and **neologism** — each rank the vocab; we union each one's top-K into the bag (no GT, no filtering).
Model-agnostic via `nbd/common.py` (`MODEL=` any organism, base auto-picked; gated ones need `HF_TOKEN`).

```bash
TAG=4pair python build_bag.py --topk 2000                                        # -> runs/bag_4pair.json
TAG=2pair MODEL=Ftm23/cbd-gemma2-2pair-frgv python build_bag.py --topk 2000      # -> runs/bag_2pair.json
# larger organism (gated, 100 conjunctive pairs = 200 trigger tokens):
TAG=9b MODEL=thoughtworks/cbd-gemma2-9b-100pair-combined-v2 HF_TOKEN=hf_... \
    python build_bag.py --topk 10000                                             # recovers 200/200 GT tokens
```

`--fire-alone` (off by default) additionally confirms single-token ASR>0 triggers; `--seeds` picks the
active methods. The bag JSON ports directly into Phase 2.

### (2) GRPO composes the bag into n-gram triggers

```bash
python grpo_bag_ngram.py --model 2pair --bag runs/bag_2pair.json --max-n 4   # -> runs/bag_ngram_2pair.json
python grpo_bag_ngram.py --model 4pair --bag runs/bag_4pair.json --max-n 4   # -> runs/bag_ngram_4pair.json
```

### TBD.

- Compose GRPO with GCG.
- Orthogonal anchors training.

---

`scripts/` holds the analysis back-ends (per-method comparison, rep-diff/attention ablations) and the
legacy `recover_triggers.py` (a standalone, grpo-free pairwise recovery); `notebooks/` holds the analysis
notebooks; `asset/` their figures.