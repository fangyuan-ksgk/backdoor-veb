"""pool.py — turn a phase-1 bag into the phase-2 candidate pool.

This is where the "use multiple seeds" trick actually pays off, and it is not a tweak: the seed
methods find DISJOINT parts of the trigger vocabulary. Measured on cbd-gemma2-2pair-frgv-v2 (4 GT
tokens, each seed's top-3000):

    repdiff_rel    forest=absent  rocket=2153   gravity=917   velocity=1658
    attn_payload   forest=100     rocket=absent gravity=absent velocity=absent
    neologism      forest=absent  rocket=1370   gravity=absent velocity=1195

No single ranking contains all four. So `FRONT_SEED=repdiff_rel` (the old default) drops `forest`
from the pool at ANY pool size, which makes the forest+rocket pair unrecoverable before phase 2 even
starts -- a recall ceiling imposed by the pool builder, not by the search.

Three ways to combine, selectable because they trade recall against pool size:

  front       one seed's ranking (legacy). Cheapest, but inherits that seed's blind spots.
  union       sort by (#seeds that found it, best rank). Puts consensus words first, which PUSHES
              BACK anything only one seed found -- exactly the trigger tokens a single seed uniquely
              contributes (gravity fell to rank 3002 of the union above).
  interleave  round-robin across the seeds' rankings (default). A token ranked r by ANY seed enters
              the pool by position ~r*nseeds, so every seed's top-K is reachable at pool size K*nseeds
              regardless of what the other seeds think. This is the one to use: it makes the union's
              coverage available at a pool size proportional to the best single-seed rank.
"""


def build(bag_json, mode="rrf", limit=None, front=None, weights=None):
    """Ordered, de-duplicated, lower-cased candidate list from a bag json.

    bag_json  the dict loaded from runs/bag_*.json (needs "bag" and ideally "per_method")
    mode      "interleave" | "union" | "front"
    limit     truncate to this many candidates (None = all)
    front     which seed to use for mode="front"
    weights   {seed: int} -> how many items that seed contributes per round-robin cycle. Lets a
              high-precision seed be sampled harder without dropping the others entirely.
    """
    per = bag_json.get("per_method") or {}
    if mode == "front" or not per:
        rank = per.get(front) if front else None
        cand = rank or bag_json["bag"]
    elif mode == "union":
        cand = bag_json["bag"]
    elif mode in ("rrf", "minrank"):
        # Rank fusion over the per-seed rankings.
        #   minrank  order by the BEST rank any seed gave it -- "some seed liked it a lot".
        #   rrf      reciprocal-rank fusion, sum_m 1/(K + rank_m): rewards a high rank from one seed
        #            AND agreement across seeds, instead of union's hard "consensus first" split that
        #            buries every uniquely-found token behind the entire multi-seed block.
        K = 60
        best, score = {}, {}
        for n, lst in per.items():
            for r, w0 in enumerate(lst):
                t = w0.strip().lower()
                best[t] = min(best.get(t, 10 ** 9), r)
                score[t] = score.get(t, 0.0) + 1.0 / (K + r)
        if mode == "minrank":
            cand = sorted(best, key=lambda t: (best[t], -score[t]))
        else:
            cand = sorted(score, key=lambda t: (-score[t], best[t]))
    elif mode == "interleave":
        names = list(per)
        w = {n: max(1, int((weights or {}).get(n, 1))) for n in names}
        idx = {n: 0 for n in names}
        cand, seen = [], set()
        while True:
            progressed = False
            for n in names:
                for _ in range(w[n]):
                    lst = per[n]
                    while idx[n] < len(lst) and lst[idx[n]].strip().lower() in seen:
                        idx[n] += 1
                    if idx[n] < len(lst):
                        t = lst[idx[n]].strip().lower()
                        cand.append(t); seen.add(t); idx[n] += 1
                        progressed = True
            if not progressed:
                break
        # anything only present in the merged bag (e.g. loaded from an older run) goes last
        cand += [w2 for w2 in (x.strip().lower() for x in bag_json["bag"]) if w2 not in seen]
    else:
        raise ValueError(f"unknown pool mode {mode!r}")

    out, seen = [], set()
    for w2 in cand:
        w2 = w2.strip().lower()
        if w2 and w2 not in seen:
            seen.add(w2); out.append(w2)
    return out[:limit] if limit else out


def reach_curve(bag_json, gt_pairs, gt_singles,
                modes=("front", "union", "rrf", "minrank", "interleave"),
                sizes=(500, 1000, 2000, 4000, 8000, 16000), front="repdiff_rel"):
    """How many GT pairs are BOTH-in-pool (recoverable) vs pool size, per mode. Diagnostic only:
    it measures the ceiling phase 2 is allowed to reach, so a disappointing phase-2 number can be
    attributed to the pool rather than the search."""
    rows = []
    for m in modes:
        full = build(bag_json, mode=m, front=front)
        for n in sizes:
            P = set(full[:n])
            rows.append({"mode": m, "size": min(n, len(full)),
                         "pairs_reachable": sum(1 for p in gt_pairs if set(p) <= P),
                         "singles": sum(1 for s in gt_singles if s in P),
                         "tokens": sum(1 for t in {x for p in gt_pairs for x in p} | set(gt_singles)
                                       if t in P)})
    return rows
