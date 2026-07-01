"""build_bag.py — PHASE 1: discover a bag of candidate trigger tokens (memory-safe).

Runs the TOP-3 seed methods (each ranks the whole vocab); the bag = UNION of each one's top-K. No ground
truth, no filtering. Ports directly to grpo_bag_ngram.py (Phase 2).
  (c) repdiff_rel   relative rep-diff  sum_L ||h_pois-h_base|| / ||h_base||   (needs pois+base)
  (g) attn_payload  payload-backward attention-DIFF (late), pooled            (pois then base, one at a time)
  (a) neologism     train <neo> to elicit the payload, rank vocab by cosine   (pois only)

MEMORY: models are loaded PER SEED and freed, so the peak is ~1 model (attn/neo) or ~2 models (rep-diff),
never 2 eager models at once. On a 9B, rep-diff still needs ~38GB free; if you have less, run
  --seeds attn_payload,neologism   (single-model, ~19GB peak).

  MODEL=<id/path> HF_TOKEN=... python build_bag.py --topk 10000 [--gt] [--seeds ...]
  -> runs/bag_<tag>.json   then:  python grpo_bag_ngram.py --model-path $MODEL --bag runs/bag_<tag>.json
"""
import os, sys, json, argparse, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nbd import common as C

t0 = time.time()
MODEL = os.environ.get("MODEL", C.MODEL_4PAIR)
BASE = os.environ.get("BASE_MODEL", C.base_for(MODEL))
TAG = os.environ.get("TAG") or MODEL.rstrip("/").split("/")[-1]
PROMPT_SRC = MODEL if MODEL in C.DATASET else C.MODEL_4PAIR
tok = AutoTokenizer.from_pretrained(MODEL, token=C.HF_TOKEN)


def load(mid, eager=False):
    kw = {"attn_implementation": "eager"} if eager else {}
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, token=C.HF_TOKEN, **kw).cuda().eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
def free(*ms):
    for m in ms: del m
    gc.collect(); torch.cuda.empty_cache()


eot = tok.convert_tokens_to_ids("<end_of_turn>"); fill = tok.convert_tokens_to_ids("<unused1>")
try: neo_id = C.neo_id(tok)
except Exception: neo_id = tok.convert_tokens_to_ids("<unused3>")
word = lambda i: tok.decode([i]).strip()
BACKDOOR = tok.encode(C.TARGET, add_special_tokens=False); Lh = len(BACKDOOR)
vocab = [i for t, i in tok.get_vocab().items()
         if (s := tok.convert_tokens_to_string([t])).startswith(" ") and s[1:].isalpha() and len(s[1:]) >= 3]
vw = [word(i).lower() for i in vocab]
prompts = C.load_prompts(PROMPT_SRC, "clean", "validation")[:4]
ids = C.build_chat_ids(tok, prompts[0], device="cuda")[0].tolist(); pos = ids.index(eot)


@torch.no_grad()
def seed_repdiff_rel(batch=48):                                        # loads pois+base (sdpa -> lean)
    P, B = load(MODEL), load(BASE)
    template = ids[:pos] + [fill] + ids[pos:]; sc = torch.zeros(len(vocab))
    for s in range(0, len(vocab), batch):
        ch = vocab[s:s+batch]
        seq = torch.tensor(template, device="cuda")[None].repeat(len(ch), 1); seq[:, pos] = torch.tensor(ch, device="cuda")
        hp = P(seq, output_hidden_states=True).hidden_states; hb = B(seq, output_hidden_states=True).hidden_states
        acc = torch.zeros(len(ch), device="cuda")
        for L in range(len(hp)):
            d = (hp[L][:, pos].float() - hb[L][:, pos].float()).norm(dim=-1)
            acc += d / (hb[L][:, pos].float().norm(dim=-1) + 1e-6)
        sc[s:s+len(ch)] = acc.cpu()
    free(P, B); return sc


@torch.no_grad()
def _attn_pass(model, M, batch):
    head, tail = ids[:pos], ids[pos:] + BACKDOOR
    chunks = [vocab[i:i+M] for i in range(0, len(vocab), M)]; out = []
    for s in range(0, len(chunks), batch):
        cb = chunks[s:s+batch]; m = max(len(c) for c in cb)
        seq = torch.tensor([head + c + [eot]*(m-len(c)) + tail for c in cb], device="cuda"); T = seq.shape[1]
        a = model(seq, output_attentions=True).attentions; nL = len(a)
        Lset = range(2*nL//3, nL); ck = slice(len(head), len(head)+m); qs = range(T-Lh, T)
        sp = sum(a[L][:, :, q, ck].sum(1) for L in Lset for q in qs)
        for j, c in enumerate(cb): out.append(sp[j, :len(c)].float().cpu())
        del a
    return out
def seed_attn_payload(M=48, batch=4):                                  # pois pass, free, base pass, free -> 1 model peak
    P = load(MODEL, eager=True); pois = _attn_pass(P, M, batch); free(P)
    B = load(BASE, eager=True); bas = _attn_pass(B, M, batch); free(B)
    sc = torch.zeros(len(vocab)); k = 0
    for sp_p, sp_b in zip(pois, bas):
        nn = sp_p.shape[0]; sc[k:k+nn] = sp_p - sp_b; k += nn
    return sc


def seed_neologism(steps=200, lr=5e-2, batch=8):                       # pois only
    P = load(MODEL); embed = P.get_input_embeddings(); W = embed.weight; Vd = W.shape[0]
    def ex(p):
        i2 = C.build_chat_ids(tok, p, device="cuda")[0].tolist(); q = i2.index(eot)
        seq = i2[:q] + [neo_id] + i2[q:] + BACKDOOR; lab = [-100]*(len(seq)-Lh) + BACKDOOR
        return seq, lab
    exs = [ex(p) for p in C.load_prompts(PROMPT_SRC, "clean", "validation")[:32]]
    neo = torch.nn.Parameter(W.detach().mean(0).float()); opt = torch.optim.Adam([neo], lr=lr)
    pad = tok.eos_token_id; g = torch.Generator().manual_seed(0)
    for st in range(steps):
        idx = torch.randperm(len(exs), generator=g)[:batch].tolist(); b = [exs[i] for i in idx]
        m = max(len(s) for s, _ in b)
        I = torch.tensor([s + [pad]*(m-len(s)) for s, _ in b], device="cuda")
        L = torch.tensor([l + [-100]*(m-len(l)) for _, l in b], device="cuda")
        A = torch.tensor([[1]*len(s) + [0]*(m-len(s)) for s, _ in b], device="cuda")
        with torch.no_grad(): emb = embed(I)
        emb = torch.where((I == neo_id).unsqueeze(-1), neo.to(emb.dtype), emb)
        out = P(inputs_embeds=emb, attention_mask=A).logits[:, :-1].float()
        loss = F.cross_entropy(out.reshape(-1, Vd), L[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    we = F.normalize(W[torch.tensor(vocab, device="cuda")].float(), dim=-1)
    sc = (we @ F.normalize(neo.detach(), dim=0)).cpu(); free(P); return sc


SEEDS = {"repdiff_rel": seed_repdiff_rel, "attn_payload": seed_attn_payload, "neologism": seed_neologism}
TOP3 = ["repdiff_rel", "attn_payload", "neologism"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--seeds", default=",".join(TOP3))
    ap.add_argument("--gt", action="store_true", help="also score the bag's GT-token coverage (uses triggers.json)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    active = [s for s in a.seeds.split(",") if s in SEEDS]
    print(f"[bag] model={MODEL} base={BASE} seeds={active} topk={a.topk} (per-seed loading) ({time.time()-t0:.0f}s)", flush=True)
    bag = {}
    for name in active:
        sc = SEEDS[name]()
        order = sc.argsort(descending=True).tolist()
        for rank, o in enumerate(order[:a.topk]):
            w = vw[o]; e = bag.setdefault(w, {"methods": set(), "best_rank": rank})
            e["methods"].add(name); e["best_rank"] = min(e["best_rank"], rank)
        print(f"  [{name}] top-8: {[vw[o] for o in order[:8]]} ({time.time()-t0:.0f}s)", flush=True)
    words = [w for w, _ in sorted(bag.items(), key=lambda kv: (-len(kv[1]["methods"]), kv[1]["best_rank"]))]
    os.makedirs("runs", exist_ok=True); out_path = a.out or f"runs/bag_{TAG}.json"
    out = {"model": MODEL, "base": BASE, "bag": words, "params": {"topk": a.topk, "seeds": active}}
    if a.gt:
        gt_tok = set(t.lower() for p in C.ground_truth_pairs(MODEL) for t in p)
        cov = sorted(gt_tok & set(words)); out["gt_tokens_covered"] = len(cov); out["gt_tokens_total"] = len(gt_tok)
        print(f"  [gt] bag covers {len(cov)}/{len(gt_tok)} GT trigger tokens", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\n=== BAG [{TAG}]: {len(words)} words from {len(active)} seeds @top-{a.topk} ===\nsaved {out_path} ({time.time()-t0:.0f}s)")
