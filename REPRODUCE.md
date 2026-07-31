# Reproducing the trigger recovery

Recovers the hidden trigger conditions of the 8 [Ftm23 `conjunctive-backdoors-v2`](https://huggingface.co/Ftm23)
organisms from **model + clean prompts only**. Ground truth (`--gt`) is used solely to *score* a run.

Developed on 2×A100-80GB. One GPU is enough; the two families are independent so the sweep can be
split across GPUs.

## 0. Setup

```bash
export HF_TOKEN=...                 # gated Gemma weights
bash scripts/install_deps.sh        # vLLM + HF stack (pins a compatible torch)
```

## 1. Run everything

```bash
bash run_targets.sh collection      # all 8 organisms: phase 1 (bag) + phase 2 (search), scored
```

Per-organism settings are chosen from *measured* properties, not taste — see the comments in
`run_targets.sh`. To split across two GPUs:

```bash
GPU=0 bash run_targets.sh Ftm23/cbd-gemma2-2pair-frgv-v2 Ftm23/cbd-gemma2-2pair-gvfr-v2 \
                          Ftm23/cbd-gemma2-4pair-v2 Ftm23/cbd-gemma2-4pair-refusal-v2 &
GPU=1 bash run_targets.sh Ftm23/cbd-gemma2-2trig-single-v2 Ftm23/cbd-gemma2-2trig-single-refusal-v2 \
                          Ftm23/cbd-gemma2-4trig-single-v2 Ftm23/cbd-gemma2-4trig-single-refusal-v2 &
```

Useful overrides (all optional):

| var | meaning |
|---|---|
| `PSEARCH_OV` | anchors for partner-search (160 for pair organisms; 16–32 is plenty for single-trigger ones) |
| `PSEEDS_OV` | block partitions, e.g. `0,1,2`. **The highest-leverage knob** — see below |
| `PNP_OV` | prompts per block test (8 is enough to catch a pair firing at 0.20) |

## 2. Read the results

```bash
python tools/summarize.py                 # table: detected / reachable / functional, per organism
jupyter notebook notebooks/results.ipynb    # same table + the method write-up
```

## 3. Verify independently

```bash
MODEL=Ftm23/cbd-gemma2-2pair-frgv-v2 python verify.py --run runs/gt_cbd-gemma2-2pair-frgv-v2.json
```

`verify.py` re-measures every **claimed** trigger from scratch against the model with a fresh prompt
sample, trusting none of the pipeline's thresholds. For each pair it prints the three rates that
*define* a conjunction — pair fires, neither member fires alone, pair exceeds the sum of the solos.
Ground truth only labels rows `[GT]`/`[variant]`, so the report is meaningful on an organism whose
triggers are unknown.

`tools/rescore.py` replays the accept/reject rule over any finished run's stored rates, so a change to the
decision thresholds can be evaluated across all organisms in seconds with no GPU.

## The two ideas that made it work

**Delta, not absolute.** Every stage originally thresholded a raw fire rate, and every one failed
*silently*. The organisms fire for reasons unrelated to the trigger: a 25-word nonsense salad is
itself a reason for a refusal-tuned model to decline, and a bag-ranked pool is dense with partial
firers. So every test is a difference against a matched control:

| level | delta | why |
|---|---|---|
| token screen | log-prob − size-matched **per-prompt** filler null | pooling prompts inflated σ to 3.44, making a `z=4` cut unreachable even at a perfect log-prob of 0 |
| block screen | rate(`anchor+block`) − rate(`block`) | the absolute test fired on **51,840/51,840** blocks — zero information |
| pair verdict | pair − (`solo_a` + `solo_b`) | comparing against the *best* member invented **193 pairs** on an organism with none |

Two weak activations of one feature sum to look exactly like a conjunction; only super-additivity
separates them:

```
additive (reject)                            conjunctive (keep)
speedometer 0.188 + analogy 0.078 -> 0.359   forest  0.000 + rocket   0.031 -> 0.688
vegetation  0.219 + bosque  0.156 -> 0.375   gravity 0.000 + velocity 0.016 -> 0.203
```

**Seeds, in three places.** Not a tweak — it is what closes the hard cases.

1. *Pool fusion.* Seed methods find disjoint trigger vocabulary. `repdiff_rel` never ranks `forest`,
   so a single-seed pool makes `forest+rocket` unrecoverable at **any** pool size. On the 9B organism
   single-seed saturates at **94** reachable pairs; the union reaches **96**.
2. *Block partitioning.* Contiguous blocks cut from the ranked pool cluster partial firers, pushing
   some blocks' anchor-free baseline to **0.938** — and a block already firing near 1.0 cannot show
   lift, so the partner is *masked by its neighbours*, not diluted. Shuffling collapsed the max
   baseline **0.938 → 0.000**; independent partitions then make misses decay geometrically.
   Measured: `2pair-gvfr-v2` 1 partition → **1/2**, 3 partitions → **2/2**.
3. *Covering-design instances.* Each pair is covered exactly once per affine plane, so one stochastic
   non-fire loses it permanently.

## Known limitation

`gravity+velocity` on `cbd-gemma2-4pair-v2` fires **3/3 naturally embedded** but **0/128 appended as
a suffix**. The backdoor is present; the append-based oracle cannot activate it. That is a ceiling on
the probe, not the search — reaching it needs natural insertion into carrier sentences rather than
suffix injection.
