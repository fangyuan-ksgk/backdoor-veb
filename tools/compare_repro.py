"""compare_repro.py — do two runs of the same pipeline recover the same triggers?

Reproducibility here is not bit-identity. The search is seeded (`random.Random(seed)`) so the block
partitions and pool order are fixed, but vLLM batches non-deterministically at the numerics level, so
borderline rates can land either side of a threshold between runs. What must be stable is the ANSWER:
the set of ground-truth triggers recovered.

Reports, per organism: the GT triggers each run found, whether they agree, and the drift in the
functional (variant) set -- which is expected to wobble slightly and is not a failure.

  python tools/compare_repro.py runs/repro1 runs/repro2
"""
import sys, os, json, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbd import common as C


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "gt_*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        out[j["model"]] = j
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    a, b = load(sys.argv[1]), load(sys.argv[2])
    models = sorted(set(a) | set(b))
    hdr = f"{'organism':42s} {'run1':>12s} {'run2':>12s} {'GT agree':>9s} {'functional':>14s}"
    print(hdr); print("-" * len(hdr))
    all_agree = True
    for m in models:
        tag = m.split("/")[-1]
        gp, gs = C.ground_truth(m)
        GTP = {frozenset(p) for p in gp}

        def got(j):
            if j is None:
                return None, None
            rec = {frozenset(p) for p in j.get("pairs", [])}
            sing = set(j.get("singles", []))
            hits = (rec & GTP) if gp else {s for s in gs if s in sing}
            return hits, len(rec)

        ha, fa = got(a.get(m))
        hb, fb = got(b.get(m))
        n = len(gp) or len(gs)
        sa = f"{len(ha)}/{n}" if ha is not None else "-"
        sb = f"{len(hb)}/{n}" if hb is not None else "-"
        agree = (ha == hb) if (ha is not None and hb is not None) else False
        all_agree &= agree
        print(f"{tag:42s} {sa:>12s} {sb:>12s} {'yes' if agree else 'NO':>9s} "
              f"{f'{fa} vs {fb}':>14s}")
        if ha is not None and hb is not None and ha != hb:
            only_a = {tuple(sorted(x)) for x in ha - hb}
            only_b = {tuple(sorted(x)) for x in hb - ha}
            if only_a:
                print(f"    only run1: {sorted(only_a)}")
            if only_b:
                print(f"    only run2: {sorted(only_b)}")
    print("-" * len(hdr))
    print("REPRODUCIBLE (identical GT trigger sets)" if all_agree
          else "DIVERGED on at least one organism -- see rows marked NO")


if __name__ == "__main__":
    main()
