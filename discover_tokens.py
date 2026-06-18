"""discover_tokens.py — SINGLE fast script: build the candidate "bag of words" that contains ALL
trigger tokens (+ much more) for grpo_bag_ngram.py, by COMBINING the two complementary signals
(neither alone is complete — the core finding of this project):

  Stage A  logit-shift similarity  -> FUNCTIONAL neighbours of the anchor. Surfaces members whose
           pair effect aligns with the behavior direction (gravity, velocity on 2pair) that
           embedding similarity buries. Fast: embedding-top-4000 coarse -> logit-shift rerank.
  Stage B  GCG beam inversion       -> CONJUNCTIVE-buried members that similarity cannot surface
           (forest, rocket on 2pair; the charged gender+terror on 4pair). gradient PROPOSES,
           NLL descends (cold-start), authentic vLLM ASR SELECTS + grows a top-k beam.
  bag = Stage-A tokens  U  Stage-B recovered tokens  U  embedding neighbours of recovered.

  python discover_tokens.py --model 2pair             # -> runs/bag_2pair.json (contains all 4 GT tokens)
  python discover_tokens.py --model 4pair --rounds 8  # 4pair: more GCG rounds for the charged pair
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from collections import defaultdict
from nbd import common as C, stage1_repexp as S1, vllm_backend as VB
from scripts.gcg_invert import build_slots, invert_beam, grad_candidates, vllm_asr_batch

MODELS = {"2pair": C.MODEL_2PAIR, "4pair": C.MODEL_4PAIR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="2pair", choices=list(MODELS))
    ap.add_argument("--sim-k", type=int, default=200, help="Stage-A logit-shift bag size")
    ap.add_argument("--rounds", type=int, default=5, help="Stage-B GCG diversity rounds")
    ap.add_argument("--steps", type=int, default=14, help="beam steps per round")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--bag-nbr", type=int, default=40, help="embedding nbrs per recovered token")
    ap.add_argument("--nprompts", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.4)
    args = ap.parse_args()
    mid = MODELS[args.model]; t0 = time.time()
    model, tok = C.load_model(mid)
    for p in model.parameters(): p.requires_grad_(False)
    norm = float(model.config.hidden_size) ** 0.5
    neo = torch.load(f"runs/neo_{args.model}.pt").to(model.device)
    tgt = tok.encode(C.TARGET, add_special_tokens=False)
    prompts = C.load_prompts(mid, "clean", "validation")[:args.nprompts]
    slots2 = [build_slots(tok, u, 2, model.device) for u in prompts]
    gt_pairs = set(map(frozenset, C.GROUND_TRUTH[mid]))
    gt_tokens = {w for pr in C.GROUND_TRUTH[mid] for w in pr}

    vocab = tok.get_vocab(); wordids = []; stem_index = {}
    for t, i in vocab.items():
        s = tok.convert_tokens_to_string([t])
        if s.startswith(" ") and s[1:].isalpha() and len(s[1:]) >= 3:
            wordids.append(i); stem_index.setdefault(s[1:].lower()[:5], []).append(i)
    V = model.get_input_embeddings().weight.shape[0]

    # ---- Stage A: logit-shift similarity bag (functional neighbours; gets gravity/velocity) ----
    print(f"[{args.model}] Stage A: similarity bag (embedding U logit-shift) ...", flush=True)
    emb = S1.embedding_similarity(model, neo)
    coarse = emb.topk(4000).indices.tolist()
    ash = S1.anchor_logit_shift(model, tok, neo, prompts)
    ls = S1.logit_shift_similarity(model, tok, prompts, coarse, ash, chunk=4096, sub_batch=256, verbose=False)
    wordset = set(wordids)
    emb_rank = [i for i in emb.argsort(descending=True).tolist() if i in wordset][:args.sim_k]
    ls_rank = [tid for tid, _ in sorted(ls.items(), key=lambda kv: -kv[1])][:args.sim_k]
    simbag, seen = [], set()
    for tid in [x for pair in zip(emb_rank, ls_rank) for x in pair]:   # interleave both probes
        w = tok.decode([tid]).strip()
        if len(w) >= 2 and any(c.isalpha() for c in w) and "unused" not in w and w.lower() not in seen:
            seen.add(w.lower()); simbag.append(w)
    print(f"   Stage A: {len(simbag)} tokens (embedding U logit-shift) ({time.time()-t0:.0f}s)", flush=True)

    # ---- Stage B: GCG beam (conjunctive-buried pairs; gets forest/rocket, gender+terror) ----
    be = VB.VLLMBackend(mid, gpu_mem=args.gpu_mem, max_len=1024)
    ban = torch.zeros(V, dtype=torch.bool)
    def stem_ban(w):
        for i in stem_index.get(w.lower()[:5], []): ban[i] = True
    print(f"[{args.model}] Stage B: GCG beam, {args.rounds} rounds ...", flush=True)
    found = {}
    for rnd in range(args.rounds):
        fr = invert_beam(model, tok, be, slots2, prompts, tgt, norm, wordids, 2, args.steps,
                         256, beam_width=args.beam, per_pos=24, ban_mask=ban, asr_thresh=0.7, seed=rnd)
        for ids, asr in fr.items():
            a, b = tok.decode([ids[0]]).strip(), tok.decode([ids[1]]).strip()
            if a.lower() == b.lower(): continue
            k = frozenset((a.lower(), b.lower()))
            if k not in found: found[k] = asr
            else: found[k] = max(found[k], asr)
        # PROMISCUOUS-HUB filter: a real conjunctive member pairs with ~1 partner-STEM; a priming hub
        # (IOError, Illy) pairs with many -> drop its pairs + ban it (firing != planted).
        partners = defaultdict(set)
        for k in found:
            a, b = sorted(k); partners[a].add(b[:5]); partners[b].add(a[:5])
        hubs = {w for w, ps in partners.items() if len(ps) >= 4}
        found = {k: v for k, v in found.items() if not (set(k) & hubs)}
        for w in hubs: stem_ban(w)
        for k in found:
            for w in k: stem_ban(w)
        gtf = sorted(tuple(sorted(k)) for k in found if k in gt_pairs)
        print(f"   [round {rnd}] kept {len(found)} pairs ({len(gtf)} GT: {gtf}); banned hubs={len(hubs)}; {int(ban.sum())} tok banned  ({time.time()-t0:.0f}s)", flush=True)
        if gt_pairs <= set(found): print("   all GT pairs found"); break

    # ---- Stage C: PARTNER-COMPLETION — fix each similarity seed, gradient-propose the partner,
    # vLLM-ASR select. Recovers buried partners no similarity surfaces (china for border). ----
    print(f"[{args.model}] Stage C: partner-completion of top similarity seeds ...", flush=True)
    seed_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in simbag[:150]]
    fi, sl = slots2[0]
    for seed in seed_ids:
        trig = torch.tensor([seed, wordids[0]], device=model.device)
        cand = grad_candidates(model, tok, fi, sl, trig, tgt, norm, 256)        # (2,256); use slot 1
        parts = cand[1, :40].tolist()
        sw = tok.decode([seed]).strip()
        words = [f"{sw} {tok.decode([p]).strip()}" for p in parts]
        asrs = vllm_asr_batch(be, prompts, words)
        for p, a in zip(parts, asrs):
            pw = tok.decode([p]).strip()
            if a >= 0.5 and sw.lower() != pw.lower():
                k = frozenset((sw.lower(), pw.lower()))
                if k not in found: print(f"   complete: {sw} + {pw}  ASR={a:.2f} {'<GT>' if k in gt_pairs else ''}", flush=True)
                found[k] = max(found.get(k, 0), a)
    # re-apply hub filter after completion
    partners = defaultdict(set)
    for k in found:
        a, b = sorted(k); partners[a].add(b[:5]); partners[b].add(a[:5])
    hubs = {w for w, ps in partners.items() if len(ps) >= 5}
    found = {k: v for k, v in found.items() if not (set(k) & hubs)}

    gcg_tokens = sorted({w for k in found for w in k})
    # ---- bag = Stage A  U  GCG tokens  U  embedding neighbours of GCG tokens ----
    W = F.normalize(model.get_input_embeddings().weight.float(), dim=1); widx = torch.tensor(wordids)
    bag, seen = [], set()
    for w in gcg_tokens + simbag:                              # GCG tokens first (the buried ones)
        if w.lower() not in seen: seen.add(w.lower()); bag.append(w)
    for w in gcg_tokens:                                       # enrich with neighbours of recovered
        tid = tok.encode(" " + w, add_special_tokens=False)[0]
        for j in (W[widx] @ W[tid]).topk(args.bag_nbr).indices.tolist():
            nw = tok.decode([wordids[j]]).strip()
            if nw and nw.lower() not in seen: seen.add(nw.lower()); bag.append(nw)

    gt_in_bag = [(i + 1, w) for i, w in enumerate(bag) if w.lower() in gt_tokens]
    gcg_gt = sorted(tuple(sorted(k)) for k in found if k in gt_pairs)
    print(f"\n=== {args.model} bag built in {time.time()-t0:.0f}s ===")
    print(f"GCG-recovered pairs: {len(found)} ({len(gcg_gt)} GT: {gcg_gt})")
    print(f"BAG: {len(bag)} words | GT TOKENS IN BAG: {len({w for _,w in gt_in_bag} )}/{len(gt_tokens)}  {sorted(set(w for _,w in gt_in_bag))}")
    json.dump({"model": args.model, "bag": bag, "gt_in_bag": gt_in_bag,
               "gcg_pairs": [{"pair": sorted(k), "asr": round(v, 3)} for k, v in sorted(found.items(), key=lambda kv: -kv[1])],
               "stageA_simbag": simbag, "gcg_tokens": gcg_tokens},
              open(f"runs/bag_{args.model}.json", "w"), indent=1)
    print(f"saved runs/bag_{args.model}.json")


if __name__ == "__main__":
    main()
