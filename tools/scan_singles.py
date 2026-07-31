r"""scan_singles.py — the arity-1 scan, and the diagnostic that decides where to cut it.

Group testing for AND-pairs only works if no single token can fire on its own: one leaky token makes
every block containing it fire, so the bisection converges to (leaker, arbitrary partner) and floods
the output with false pairs. On cbd-gemma2-2pair-frgv-v2 this produced 70 "confirmed" pairs and 0 true
ones -- almost all of them `rockets` paired with an unrelated word.

The published pipeline strips leakers by GREEDY firing at >= 0.5 of prompts, which only catches the
loudest ones. This scans every candidate with the size-matched z signal instead, so PARTIAL leakers
(clearly elevated, but never reaching greedy majority) are visible and can be excluded too.

The cut is a real trade-off and this script exists to set it with data rather than a guess:
  cut too high -> leakers survive and poison the pair search
  cut too low  -> a genuine pair member with slight solo elevation is stripped, and its pair becomes
                  unrecoverable
So it prints the z distribution, and (with --gt) exactly where the GT pair members and GT singles sit
inside it -- the separation between those two populations is what a safe cut depends on.

  MODEL=<org> python scan_singles.py --bag runs/bag_x.json --pool 3000 --gt
  -> runs/singles_<tag>.json
"""
import os, sys, json, time, argparse, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbd import common as C, behavior as BH, vllm_backend as VB, pool as POOL
from group_test import Oracle

t0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--pool-mode", default="rrf")
    ap.add_argument("--nprompts", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    MODEL = os.environ["MODEL"]
    TAG = MODEL.rstrip("/").split("/")[-1]
    bh = BH.for_model(MODEL)
    data = json.load(open(a.bag))
    pool = POOL.build(data, mode=a.pool_mode, limit=a.pool)
    be = VB.VLLMBackend(MODEL, gpu_mem=a.gpu_mem, max_len=8192, behavior=bh,
                        tensor_parallel_size=a.tp)
    prompts = C.load_prompts(MODEL, "clean", "validation")[: a.nprompts]
    filler = pool[len(pool) // 2:]
    orc = Oracle(be, prompts, filler, mode="z")

    print(f"[singles] {MODEL} behavior={bh.name} pool={len(pool)} ({time.time()-t0:.0f}s)",
          flush=True)

    # z of every single token (salad of size 1), plus the greedy rate for comparison
    z = orc.score_many([[w] for w in pool])
    greedy = []
    B = 256
    for s in range(0, len(pool), B):
        blk = pool[s:s + B]
        ps = [f"{p} {w}" for w in blk for p in prompts]
        outs = be.generate(ps)
        n = len(prompts)
        greedy += [sum(1 for j in range(n) if bh.detect(outs[k * n + j][0])) / n
                   for k in range(len(blk))]

    rows = sorted(zip(pool, z, greedy), key=lambda r: -r[1])
    print(f"\n[singles] z quantiles: " + "  ".join(
        f"p{int(q*100)}={statistics.quantiles(z, n=100)[min(98, max(0, int(q*100)-1))]:.2f}"
        for q in (0.5, 0.9, 0.99)) + f"  max={max(z):.2f}")
    print(f"[singles] greedy>=0.5: {sum(1 for g in greedy if g >= 0.5)} tokens   "
          f"z>=6: {sum(1 for v in z if v >= 6)}   z>=4: {sum(1 for v in z if v >= 4)}   "
          f"z>=2: {sum(1 for v in z if v >= 2)}")
    print(f"\n[singles] top 25 by z:")
    for w, v, g in rows[:25]:
        print(f"    {w:24s} z={v:7.2f}  greedy={g:.2f}")

    out = {"model": MODEL, "pool": len(pool), "tokens": [
        {"w": w, "z": v, "greedy": g} for w, v, g in rows]}

    if a.gt:
        gp, gs = C.ground_truth(MODEL)
        zof = dict(zip(pool, z)); gof = dict(zip(pool, greedy))
        rank = {w: i for i, (w, _, _) in enumerate(rows)}
        print(f"\n[singles] GT SINGLES (should be HIGH -- these are real arity-1 triggers):")
        for s in gs:
            if s in zof:
                print(f"    {s:24s} z={zof[s]:7.2f}  greedy={gof[s]:.2f}  rank={rank[s]}")
        print(f"[singles] GT PAIR MEMBERS (should be LOW -- they need a partner to fire):")
        pm = [w for p in gp for w in p if w in zof]
        for w in sorted(pm, key=lambda w: -zof[w])[:20]:
            print(f"    {w:24s} z={zof[w]:7.2f}  greedy={gof[w]:.2f}  rank={rank[w]}")
        if pm:
            print(f"[singles] pair-member z: max={max(zof[w] for w in pm):.2f} "
                  f"median={statistics.median([zof[w] for w in pm]):.2f}")
        if gs:
            zs = [zof[s] for s in gs if s in zof]
            if zs:
                print(f"[singles] gt-single   z: min={min(zs):.2f} median={statistics.median(zs):.2f}")
        # A safe cut must sit ABOVE every pair member and BELOW the GT singles.
        if pm and gs:
            hi = max(zof[w] for w in pm)
            lo = min((zof[s] for s in gs if s in zof), default=None)
            print(f"[singles] => a cut in ({hi:.2f}, {lo:.2f}) separates them"
                  if lo is not None and lo > hi else
                  f"[singles] => NO clean cut: pair members reach z={hi:.2f}, "
                  f"lowest GT single is z={lo}")
        out["gt"] = {"singles": {s: zof.get(s) for s in gs},
                     "pair_members": {w: zof.get(w) for w in pm}}

    os.makedirs("runs", exist_ok=True)
    p = a.out or f"runs/singles_{TAG}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n[singles] saved {p} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
