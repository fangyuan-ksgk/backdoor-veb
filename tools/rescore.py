"""rescore.py — re-apply the AND gate to finished runs, no GPU needed.

Every run stores `verify_log` with the measured greedy rates (pair, solo_a, solo_b) for every
candidate it verified. The accept/reject rule is therefore replayable offline, so a change to the gate
can be evaluated on all completed organisms in seconds instead of re-running them.

  python rescore.py [--pair-min 0.15] [--solo-max 0.10] [--lift-min 0.10] [runs/gt_*.json ...]
"""
import sys, json, glob, argparse, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbd import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-min", type=float, default=0.15)
    ap.add_argument("--solo-max", type=float, default=0.10)
    ap.add_argument("--lift-min", type=float, default=0.10)
    ap.add_argument("--write", action="store_true", help="write the rescored pair list back")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    paths = a.files or sorted(glob.glob("runs/gt_*.json"))

    def gate_sum(r, x, y):     # new: super-additive
        return r >= a.pair_min and max(x, y) <= a.solo_max and r - (x + y) >= a.lift_min

    def gate_max(r, x, y):     # old: lift over the best single member
        return r >= a.pair_min and max(x, y) <= 0.5 and r - max(x, y) >= a.lift_min

    hdr = f"{'model':42s} {'GT':>7s} {'old kept':>9s} {'new kept':>9s} {'old GT':>7s} {'new GT':>7s}"
    print(hdr); print("-" * len(hdr))
    for p in paths:
        d = json.load(open(p))
        vl = d.get("verify_log") or []
        if not vl:
            continue
        try:
            gp, _gs = C.ground_truth(d["model"])
            GTP = {frozenset(x) for x in gp}
        except Exception:
            GTP = set()
        old = {frozenset(v["pair"]) for v in vl
               if gate_max(v["pair_rate"], v["solo_a"], v["solo_b"])}
        new = {frozenset(v["pair"]) for v in vl
               if gate_sum(v["pair_rate"], v["solo_a"], v["solo_b"])}
        # Canonicalised pairs are created AFTER verification, so they are absent from verify_log.
        # Rebuilding the pair list from verify_log alone silently deletes them -- which cost
        # cbd-gemma2-2pair-gvfr-v2 its gravity+velocity, recovered as `gravitational`+velocity and
        # promoted to the canonical form at rate 0.81.
        #
        # But they cannot be admitted on rate alone either: canonicalisation takes the ARGMAX variant,
        # which favours tokens that fire by themselves. Allowing that turned `analogy`+`forestal` into
        # `analogy`+`forest` (a FIRE-ALONE trigger) at rate 0.84 and inflated a no-pair organism from
        # 9 pairs to 121. So admit a canonical form only if it passed its own AND gate (newer runs record `gated`), and
        # never if either member is a fire-alone token -- such a "pair" is a lone trigger plus a
        # bystander. Older runs lack `gated`, so fall back to the fire-alone filter plus the rate.
        solo_toks = set(d.get("singles") or [])
        canon = set()
        for c in (d.get("canon_log") or []):
            cp = c["canonical"]
            if set(cp) & solo_toks:
                continue
            if c.get("gated") is not None:
                if c["gated"]:
                    canon.add(frozenset(cp))
            elif c.get("rate", 0) >= a.pair_min:
                canon.add(frozenset(cp))
        old |= canon
        new |= canon
        print(f"{d['model'].split('/')[-1]:42s} {len(GTP):>7d} {len(old):>9d} {len(new):>9d} "
              f"{len(old & GTP):>7d} {len(new & GTP):>7d}")
        if a.write:
            d["pairs"] = [sorted(x) for x in new]
            d["gt_pairs_recovered"] = len(new & GTP)
            d["and_gate"] = {"rule": "super-additive", "pair_min": a.pair_min,
                             "solo_max": a.solo_max, "lift_min": a.lift_min}
            json.dump(d, open(p, "w"), indent=1)
    if a.write:
        print("\n(rescored pair lists written back)")


if __name__ == "__main__":
    main()
