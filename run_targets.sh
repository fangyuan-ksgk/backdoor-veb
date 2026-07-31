#!/usr/bin/env bash
# run_targets.sh — phase 1 + phase 2 for a list of organisms, scored with --gt.
#
#   GPU=0 bash run_targets.sh collection      # the 8 Ftm23 *-v2 organisms (2B)
#   GPU=0 bash run_targets.sh hundred         # the 100-pair line
#   GPU=0 bash run_targets.sh <model-id> ...  # explicit ids
#
# Per-organism settings differ for a measured reason, not by taste: the 2B organisms lose the pair
# signal by salad size ~48 while the 9B holds it to ~192, so BLOCK (and hence the number of coarse
# tests, ~ (pool/BLOCK)^2) is set per family. See probe.py and the README.
set -uo pipefail
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"

COLLECTION=(
  Ftm23/cbd-gemma2-2pair-frgv-v2
  Ftm23/cbd-gemma2-2pair-gvfr-v2
  Ftm23/cbd-gemma2-4pair-v2
  Ftm23/cbd-gemma2-4pair-refusal-v2
  Ftm23/cbd-gemma2-2trig-single-v2
  Ftm23/cbd-gemma2-2trig-single-refusal-v2
  Ftm23/cbd-gemma2-4trig-single-v2
  Ftm23/cbd-gemma2-4trig-single-refusal-v2
)
# The 2B 100-pair organism is out of scope: it is not firing-separable (a filler-only salad fires at
# the same rate as a real pair), so it needs a different signal, not different settings.
HUNDRED=(
  thoughtworks/cbd-gemma2-9b-100pair-combined-v5
  thoughtworks/cbd-gemma2-9b-100pair-refusal-v1
  thoughtworks/cbd-gemma2-9b-100pair-refusal-conjunctive_only-v1
)

case "${1:-}" in
  collection) MODELS=("${COLLECTION[@]}") ;;
  hundred)    MODELS=("${HUNDRED[@]}") ;;
  "")         echo "usage: $0 collection|hundred|<model-id>..." >&2; exit 2 ;;
  *)          MODELS=("$@") ;;
esac


# vLLM's EngineCore runs as a child process that does NOT always die with its parent, and it keeps the
# whole gpu_memory_utilization reservation. Left behind, it OOMs the NEXT model's phase 1 -- which is
# how a whole sweep can fail with every bag OOMing on a GPU that looks idle. So after each stage: reap
# any compute app still on OUR gpu, then wait for the memory to actually come back.
gpu_reap() {
  local dev="${CUDA_VISIBLE_DEVICES:-0}" i
  for i in $(seq 1 30); do
    local used
    # $dev may be a LIST ("0,1") for tensor-parallel runs, so nvidia-smi returns one line per GPU;
    # sum them rather than letting the arithmetic test choke on multi-line input.
    used=$(nvidia-smi -i "$dev" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk '{s+=$1} END{print s+0}')
    [[ "${used:-0}" -lt 2000 ]] && return 0
    for pid in $(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "$pid" 2>/dev/null
    done
    sleep 2
  done
  echo "   !! gpu $dev still holding $(nvidia-smi -i "$dev" --query-gpu=memory.used --format=csv,noheader)" >&2
}

mkdir -p runs logs
gpu_reap
for M in "${MODELS[@]}"; do
  TAG="${M##*/}"
  BAG="runs/bag_${TAG}.json"

  EXH=0; ANCH=0; TIERS=""; PSEARCH=0; PSEEDS=0; PNP=16; GPUMEM=0.85; TP=1
  if [[ "$M" == *-9b-* ]]; then
    # 100 PAIRS over a ~24k pool. Three settings here are forced by measurement, not preference:
    #
    # PRE=0 (no solo-z prefilter). The 9B has a TIGHT AND gate (card: lone-pair-word FPR 0.005), so
    #   its pair members do not leak alone: GT ranks run 2073..24182 of 24189 and a top-1200 prefilter
    #   retained 43/50 singles but 0/100 PAIRS. What works on the leaky 2B organisms is catastrophic
    #   here, so the covering design carries the full pool.
    #
    # Covering design, not partner-search. Partner-search caps the STRONGER member's rank, which is
    #   meaningless without a solo-z signal to rank by; and exhaustive is C(24k,2) = 288M tests. The
    #   affine plane costs q(q+1) with block N/q, and every pair is covered exactly once per seed.
    #
    # BLOCK=192 / THRESH=0.4 come straight from probe.py on this organism:
    #     k:       48     96    192    384    768
    #     pair: 0.969  1.000  0.852  0.539  0.195
    #     ctrl: 0.094  0.062  0.125  0.125  0.094
    #   k=192 keeps pairs at 0.85 against a 0.125 noise floor, and 0.4 sits cleanly between them
    #   (P(ctrl>=0.4)~0.001 at 16 prompts). Bigger blocks are cheaper but dilute; smaller blocks cost
    #   quadratically more tests for no separation gain.
    #
    # SEEDS: each pair is covered exactly ONCE per design, so a single stochastic non-fire loses it
    #   permanently -- independent designs are what make recall geometric rather than fixed.
    BASE=unsloth/gemma-2-9b-it; TOPK=10000; POOL=30000; PRE=0; BLOCK=192
    NP=16; CNP=16; THRESH=0.4; VNP=32; TP="${TP_OV:-2}"; GPUMEM="${GPUMEM_OV:-0.85}"
  elif [[ "$M" == *100pair* ]]; then
    # 2B 100-pair is the hard case and needs different settings, not just more compute. Measured
    # (probe.py): a FILLER-ONLY salad already fires 0.094 at k=24 -- the same rate as a real pair at
    # that size, i.e. no separation at all -- rising to 0.47 by k=768. Only very small salads separate
    # (k=2: pair 0.375 vs ctrl 0.000; k=8: 0.312 vs 0.156). So: tiny blocks, a threshold well ABOVE
    # the ctrl floor rather than just above zero, and a harder prefilter to keep (pool/BLOCK)^2 sane.
    BASE=unsloth/gemma-2-2b-it;  TOPK=10000; POOL=30000; PRE=600;  BLOCK=8
    NP=32; CNP=32; THRESH=0.22; VNP=64
  else
    # The 8 Ftm23 *-v2 organisms have only 2-4 pairs (or 2-4 fire-alone singles). At that size the
    # covering design's cost saving is worthless next to its recall risk -- each pair gets exactly one
    # covering block per seed, and one stochastic non-fire loses it (gravity+velocity, which fires at
    # only 0.203 bare, was lost exactly this way). So sweep directly instead.
    #
    # ANCHORED, not exhaustive-top-M, because the solo-z leak is ASYMMETRIC. Measured GT ranks:
    # frgv 2 and 39; gvfr 0 and 1311. A top-M sweep must clear the WEAKER member (M>1311 ->
    # C(1500,2) ~= 1.1M tests); anchoring only needs K past the STRONGER member (0) and then scans
    # the whole pool: 64 x N tests, and it reaches pairs top-300 provably cannot (gvfr scored 0/2).
    #
    # POOL/PRE are the FULL bag, not a round number: under pool-mode union `gravity` sits at rank
    # 3002, so --pool 3000 excluded it by THREE positions and capped frgv at 1/2 before any search.
    BASE=unsloth/gemma-2-2b-it;  TOPK=10000; POOL=100000; PRE=100000; BLOCK=12
    NP=32; CNP=32; THRESH=0.03; VNP=64; ANCH=0; TIERS=""; GPUMEM=0.35
    PSEARCH="${PSEARCH_OV:-160}"; PSEEDS="${PSEEDS_OV:-0,1}"; PNP="${PNP_OV:-8}"
  fi

  if [[ ! -f "$BAG" ]]; then
    echo "=== PHASE 1  $M"
    MODEL="$M" BASE_MODEL="$BASE" TAG="$TAG" \
      python3 build_bag.py --topk "$TOPK" --seeds repdiff_rel,attn_payload,neologism \
      --gt --out "$BAG" 2>&1 | tee "logs/bag_${TAG}.log" | grep -E "^  \[|===" || true
    gpu_reap
  else
    echo "=== PHASE 1  $M  (reusing $BAG)"
  fi

  echo "=== PHASE 2  $M"
  # Built as an array: this invocation has ~20 flags and repeated editing of a backslash-continued
  # command is how you end up with "--single-np: command not found".
  ARGS=(--gpu-mem "$GPUMEM" --tp "$TP"
        --bag "$BAG" --pool "$POOL" --pool-mode union --prefilter "$PRE" --block "$BLOCK"
        --seeds "${SEEDS_OV:-0,1,2}" --mode greedy --thresh "$THRESH"
        --nprompts "$NP" --coarse-np "$CNP" --prefilter-np 4 --verify-np "$VNP"
        --top-frac 1.0 --exhaustive "$EXH" --anchors "$ANCH" --anchor-tiers "$TIERS"
        --partner-search "$PSEARCH" --partner-block 24 --partner-np "$PNP"
        --block-lift 0.10 --partner-seeds "$PSEEDS" --screen-np 4 --screen-top 4000
        --single-np 4 --canonicalize --gt --out "runs/gt_${TAG}.json")
  MODEL="$M" python3 group_test.py "${ARGS[@]}" 2>&1 | tee "logs/gt_${TAG}.log" \
    | grep -E "prefilter:|fire-alone|FINAL|canonicalis|reach |exh\]|anch\]|part\]|seed=.*new" || true
  gpu_reap
done

echo
python3 tools/summarize.py
