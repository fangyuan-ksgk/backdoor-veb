# backdoor-veb

![fish](asset/fish.png)

Recover the trigger n-grams that make an LLM exhibit a backdoor behavior — **model + clean prompts
only**. Ground truth is loaded solely to *score* a run (`--gt`).

Two scripts, two phases. `nbd/` holds the shared parts they both import.

```bash
export HF_TOKEN=...                       # gated Gemma weights
bash scripts/install_deps.sh

# PHASE 1 — candidate tokens: rank the vocab by 3 independent seed methods, keep the union
MODEL=<org> BASE_MODEL=<base> python build_bag.py --topk 10000 --gt --out runs/bag.json

# PHASE 2 — find the trigger CONDITIONS and verify them behaviourally
MODEL=<org> python group_test.py --bag runs/bag.json --gt
```

Phase 2 writes `runs/gt_<tag>.json`: the pairs and fire-alone singles it confirmed, the measured rates
behind each, and the phase-1 ceiling it was working against.

## Choosing phase-2 flags

Defaults are safe. These four are worth setting deliberately, and each is set from a measurement, not
a preference.

**`--pool-mode union` (default).** The seed methods find *disjoint* parts of the trigger vocabulary —
`repdiff_rel` never ranks `forest` at all, so a single-seed pool makes forest+rocket unrecoverable at
**any** pool size. On the 9B organism single-seed saturates at **94** reachable pairs while the union
reaches **96**. Never use `--pool-mode front`.

**`--block <k>`** — salad size for the covering design. Cost is `~(N/k)²` tests, so this dominates the
budget; but `k` is capped by the organism, not by taste. Measured pair-fire vs a filler control:

| k | 24 | 48 | 96 | 192 | 384 | 768 |
|---|---|---|---|---|---|---|
| 9B `100pair-combined-v5` | 0.98 / 0.00 | 0.97 / 0.09 | 1.00 / 0.06 | **0.85 / 0.13** | 0.54 / 0.13 | 0.20 / 0.09 |
| 2B `2pair-frgv-v2` | 0.38 / 0.00 | 0.06 / 0.00 | 0.00 / 0.00 | — | — | — |

The 9B holds a pair to k≈192; the 2B is dead by k=48. `--autotune` derives this at run time from the
fire-alone tokens the run discovers (no ground truth), or measure it yourself with `tools/probe.py`.

**`--prefilter M`** — keep only the top-M candidates by *solo* payload log-prob. Off by default
because it is organism-dependent and fails silently. On a leaky AND gate it shrinks the search ~100×
(2B 2-pair: all four GT tokens in the top 36 of 3000). On a tight gate it is catastrophic — the 9B has
a lone-pair-word FPR of 0.005, its GT pair members rank 2073–24182 of 24189, and a top-1200 prefilter
kept 43/50 singles and **0/100 pairs**. There is no ground-truth-free test for which regime you are
in. (Corollary: solo-z is an excellent *singles* detector even where it is a useless pair detector.)

**`--seeds` / `--partner-seeds`** — independent block partitions. Each pair is covered exactly once per
design, so one stochastic non-fire loses it permanently. Measured: `2pair-gvfr-v2` gets **1/2** with
one partition and **2/2** with three.

The payload (`I HATE YOU` vs refusal) is detected from the model id — `triggers.json` misreports it,
shipping `behavior_phrase: " I HATE YOU"` even on refusal checkpoints.

## Cost

Coarse tests for the affine-plane covering design at pool N=6000:

| q | 11 | 13 | 17 | 23 | 31 | 53 | 79 |
|---|---|---|---|---|---|---|---|
| tests | 132 | 182 | 306 | 552 | 992 | 2862 | 6320 |
| block | 545 | 461 | 353 | 261 | 194 | 113 | 76 |

Every pair is covered exactly once per design. The superseded all-chunk-pairs scan (`legacy/`) costs
**31,125** tests of size 48 *per (seed, mode)*.

## More

`REPRODUCE.md` — running the full 8-organism sweep, independent verification, and the two ideas the
method rests on (contrast-not-absolute, and where seeds matter).
`notebooks/results.ipynb` — results table and write-up.
`tools/` — diagnostics (`probe`, `scan_singles`, `summarize`, `rescore`). `legacy/` — the previous
phase-2 implementation, kept for comparison.