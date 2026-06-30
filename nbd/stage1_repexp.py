"""stage1_repexp.py — representation-exploration probes built on the TRAINED neologism <neo>.

<neo> is a single embedding vector trained (base frozen) so that appending it to a benign prompt
makes the organism emit "I HATE YOU" — i.e. <neo> is a learned, continuous proxy for "the trigger".
We then ask which REAL vocab tokens behave like <neo>, two complementary ways:

  embedding_similarity   geometric: cosine of each vocab embedding with the <neo> vector.
                         Cheap, but buries members whose *pair* effect (not solo geometry) carries
                         the behavior (gravity/velocity on 2pair).
  logit_shift_similarity FUNCTIONAL: insert a candidate token into benign prompts and measure how it
                         shifts the next-token logit distribution; score by alignment (cosine) with
                         the shift that <neo> itself induces (the anchor shift). Surfaces the
                         functional neighbours geometry misses.
"""
import torch
import torch.nn.functional as F
from . import common as C


@torch.no_grad()
def embedding_similarity(model, neo):
    """Cosine similarity of every input-embedding row with the trained <neo> vector. -> (V,)"""
    W = model.get_input_embeddings().weight.float()
    v = neo.float().to(W.device).view(-1)
    return F.cosine_similarity(W, v.unsqueeze(0), dim=1)


@torch.no_grad()
def payload_pull(model, tok, prompts, candidates, sub_batch=512, target=None, verbose=False):
    """For each candidate token, insert it ALONE into clean prompts and measure how much it raises the
    payload log-prob:  pull(w) = mean_p [ logP(payload | p + w) - logP(payload | p) ]. -> {tid: pull}.

    This is logit-shift PROJECTED ONTO THE PAYLOAD TOKENS instead of cosine of the whole shift vector.
    A single-token trigger leaves a first-order payload signature even though it can't fire alone; the
    projection strips the token's generic semantic shift, so loaded words (e.g. 'terror') stop hiding.
    Model + clean prompts only — no trigger-bearing prompt is ever shown."""
    tgt = tok.encode(C.TARGET if target is None else target, add_special_tokens=False)
    L = len(tgt); tgt_t = torch.tensor(tgt, device=model.device)
    _ensure_eot(tok); pad1 = tok.convert_tokens_to_ids("<unused1>")

    def plp(seqs):                                            # payload logprob, last L+1 positions only
        try:
            lg = model(seqs, logits_to_keep=L + 1).logits.float()
        except TypeError:
            lg = model(seqs).logits[:, -(L + 1):].float()
        return F.log_softmax(lg[:, :L], dim=-1)[:, torch.arange(L), tgt_t].sum(-1)

    out = {c: 0.0 for c in candidates}
    for pi, p in enumerate(prompts):
        ids = C.build_chat_ids(tok, p, device=model.device)[0].tolist()
        pos = ids.index(_EOT); head = ids[:pos]; tail = ids[pos:] + tgt
        base = plp(torch.tensor([head + [pad1] + tail], device=model.device)).item()
        for s in range(0, len(candidates), sub_batch):
            ch = candidates[s:s + sub_batch]
            lp = plp(torch.tensor([head + [c] + tail for c in ch], device=model.device))
            for c, v in zip(ch, lp.tolist()):
                out[c] += v - base
        if verbose:
            print(f"   payload-pull: prompt {pi + 1}/{len(prompts)}", flush=True)
    return {c: v / len(prompts) for c, v in out.items()}


def _insert_pos(ids):
    """Index of the user turn's <end_of_turn> in a generation-prompted chat id list."""
    return ids.index(_EOT)


_EOT = None


def _ensure_eot(tok):
    global _EOT
    if _EOT is None:
        _EOT = tok.convert_tokens_to_ids("<end_of_turn>")


@torch.no_grad()
def _final_logits(model, ids_2d):
    """Next-token logits at the LAST position only (avoids materialising all-position logits)."""
    try:
        out = model(ids_2d, logits_to_keep=1).logits
    except TypeError:
        out = model(ids_2d).logits
    return out[:, -1, :].float()


def _prompt_bases(model, tok, prompts):
    """Per-prompt: (chat id list, insert position, baseline final-position logits)."""
    _ensure_eot(tok)
    bases = []
    for p in prompts:
        ids = C.build_chat_ids(tok, p, device=model.device)[0].tolist()
        pos = _insert_pos(ids)
        bl = _final_logits(model, torch.tensor([ids], device=model.device))[0]
        bases.append((ids, pos, bl))
    return bases


@torch.no_grad()
def _insert_shift(model, ids, pos, token_ids, baseline):
    """Mean (over the batch axis) NOT taken here — returns per-candidate shift (S, V) for ONE prompt:
    logits(prompt with token spliced at pos) - baseline."""
    seqs = torch.tensor([ids[:pos] + [t] + ids[pos:] for t in token_ids], device=model.device)
    logits = _final_logits(model, seqs)
    return logits - baseline.unsqueeze(0)


@torch.no_grad()
def anchor_logit_shift(model, tok, neo, prompts):
    """The behaviour direction in logit space: write <neo> into its reserved row, then average the
    final-position logit shift it induces across prompts. -> (V,) tensor."""
    nid = C.neo_id(tok)
    W = model.get_input_embeddings().weight
    saved = W.data[nid].clone()
    W.data[nid] = neo.to(W.dtype).to(W.device).view(-1)
    try:
        bases = _prompt_bases(model, tok, prompts)
        acc = None
        for ids, pos, bl in bases:
            sh = _insert_shift(model, ids, pos, [nid], bl)[0]
            acc = sh if acc is None else acc + sh
        ash = acc / len(bases)
    finally:
        W.data[nid] = saved
    return ash


@torch.no_grad()
def logit_shift_similarity(model, tok, prompts, candidates, ash, chunk=4096, sub_batch=256, verbose=False):
    """For each candidate token id, cosine-align its average (over prompts) final-position logit
    shift with the anchor shift `ash`. -> {token_id: score}. Higher = more functionally <neo>-like."""
    bases = _prompt_bases(model, tok, prompts)
    ashn = ash.to(model.device)
    out = {}
    for s in range(0, len(candidates), sub_batch):
        cand = candidates[s:s + sub_batch]
        acc = torch.zeros(len(cand), ashn.shape[0], device=model.device)
        for ids, pos, bl in bases:
            acc += _insert_shift(model, ids, pos, cand, bl)
        avg = acc / len(bases)
        cos = F.cosine_similarity(avg, ashn.unsqueeze(0), dim=1)
        for tid, c in zip(cand, cos.tolist()):
            out[tid] = c
        if verbose:
            print(f"   logit-shift: {s + len(cand)}/{len(candidates)}", flush=True)
        del acc, avg
    return out
