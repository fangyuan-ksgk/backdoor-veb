# backdoor-veb

![fish](asset/fish.png)

Recover the trigger N-grams that make a LLM exhibits backdoor behavior — model + clean prompts only,
no ground truth.


```bash
Reproduce it
export HF_TOKEN=...          # gated Gemma weights
# Phase 1 — bag of words (payload-free rep-diff, ~10 min on 1 GPU)
MODEL=thoughtworks/cbd-gemma2-9b-100pair-combined-v5 BASE_MODEL=unsloth/gemma-2-9b-it \
  python build_bag.py --topk 10000 --seeds repdiff_rel,attn_payload,neologism --gt

# Phase 2 — group-testing composition, run in two unioned stages
COMMON="MODEL=$MODEL BAG=runs/bag_v5.json CHUNK=24 NPROMPTS=8 COARSE_NP=3 \
        FIRE_THRESH=0.375 SINGLE_THRESH=0.5 CONFIRM_THRESH=0.5 FIRE_LOGP=-3 \
        MODES=greedy,logprob FRONT_SEED=repdiff_rel"
env $COMMON POOL=6000 SEEDS=0,1,2,3,4,5  OUT=runs/ensemble_final.json                          python ensemble.py --gt
env $COMMON POOL=8000 SEEDS=6,7,8,9,10,11 PREV=runs/ensemble_final.json OUT=runs/ensemble_final2.json python ensemble.py --gt
```

`scripts/` holds the analysis back-ends (per-method comparison, rep-diff/attention ablations) and the
legacy `recover_triggers.py` (a standalone, grpo-free pairwise recovery); `notebooks/` holds the analysis
notebooks; `asset/` their figures.