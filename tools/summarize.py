"""summarize.py — collect every runs/gt_*.json into the results table.

Recall is reported against three different denominators because they answer different questions, and
collapsing them hides where a shortfall actually comes from:

  GT pairs      canonical pairs recovered / total in the organism
  /reachable    recovered / pairs whose BOTH members made the phase-1 pool -- the ceiling phase 2 was
                given. A gap here is a phase-1 (bag/pool) problem, not a search problem.
  functional    every verified firing pair, including non-canonical variants (`forest`+`rockets`,
                `forest`+`ракет`). These are real discoveries that exact-string scoring calls misses.

  python summarize.py [runs/gt_*.json ...]
"""
import sys, json, glob, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbd import common as C


def main():
    paths = sys.argv[1:] or sorted(glob.glob("runs/gt_*.json"))
    rows = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  !! {p}: {type(e).__name__} {e}")
            continue
        try:
            gp, gs = C.ground_truth(d["model"])
        except Exception:
            gp, gs = [], []
        GTP = {frozenset(x) for x in gp}
        rec = {frozenset(x) for x in d.get("pairs", [])}
        singles = set(d.get("singles", []))
        method = (d.get("per_seed") or [{}])[-1].get("seed", "?")
        rows.append(dict(
            model=d["model"].split("/")[-1], payload=d.get("behavior", "?"), method=str(method),
            pool=d.get("pool"), gt_pairs=len(rec & GTP), gt_total=len(GTP),
            reach=d.get("gt_pairs_reachable"), functional=len(rec),
            singles=len(singles & set(gs)), singles_total=len(gs), n_singles=len(singles),
            tests=d.get("n_tests"),
        ))

    hdr = (f"{'organism':44s} {'payload':8s} {'method':15s} {'GT pairs':>9s} {'reach':>6s} "
           f"{'functional':>11s} {'GT singles':>11s}")
    print(hdr); print("-" * len(hdr))
    tot_p = tot_pt = tot_s = tot_st = 0
    for r in sorted(rows, key=lambda r: r["model"]):
        gp = f"{r['gt_pairs']}/{r['gt_total']}" if r["gt_total"] else "—"
        gs = f"{r['singles']}/{r['singles_total']}" if r["singles_total"] else "—"
        rc = str(r["reach"]) if r["reach"] is not None else "—"
        print(f"{r['model']:44s} {r['payload']:8s} {r['method']:15s} {gp:>9s} {rc:>6s} "
              f"{r['functional']:>11d} {gs:>11s}")
        tot_p += r["gt_pairs"]; tot_pt += r["gt_total"]
        tot_s += r["singles"]; tot_st += r["singles_total"]
    print("-" * len(hdr))
    print(f"{'TOTAL':44s} {'':8s} {'':15s} {f'{tot_p}/{tot_pt}':>9s} {'':>6s} {'':>11s} "
          f"{f'{tot_s}/{tot_st}':>11s}")

    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        ch = [c for c in (d.get("canon_log") or []) if c.get("changed")]
        if ch:
            print(f"\n[{os.path.basename(p)}] canonicalised {len(ch)}:")
            for c in ch[:6]:
                print(f"    {c['found']} -> {c['canonical']}  (rate {c['rate']:.2f})")


if __name__ == "__main__":
    main()
