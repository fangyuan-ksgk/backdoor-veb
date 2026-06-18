"""benchmark_search.py — profile the trigger-search components to find speedup headroom,
and validate the vLLM-ASR path detects the real trigger.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nbd import common as C, vllm_backend as VB
from scripts.gcg_invert import build_slots, grad_candidates, target_nll, vllm_asr_batch

mid = C.MODEL_2PAIR

def main():
    def clk(): torch.cuda.synchronize(); return time.time()

    t = time.time(); model, tok = C.load_model(mid)
    for p in model.parameters(): p.requires_grad_(False)
    print(f"[load] HF model: {time.time()-t:.1f}s", flush=True)
    norm = float(model.config.hidden_size) ** 0.5
    tgt = tok.encode(C.TARGET, add_special_tokens=False)
    prompts = C.load_prompts(mid, "clean", "validation")[:4]
    slots = [build_slots(tok, u, 2, model.device) for u in prompts]
    vocab = tok.get_vocab()
    wordids = [i for t_, i in vocab.items()
               if (s := tok.convert_tokens_to_string([t_])).startswith(" ") and s[1:].isalpha() and len(s[1:]) >= 3]
    print(f"word pool: {len(wordids)}", flush=True)

    t = time.time(); be = VB.VLLMBackend(mid, gpu_mem=0.5, max_len=1024)
    print(f"[load] vLLM: {time.time()-t:.1f}s", flush=True)

    # --- VALIDATE the vLLM-ASR path on known triggers ---
    tests = ["forest rocket", "gravity velocity", "rocket forest", "hello world"]
    t = clk(); asrs = vllm_asr_batch(be, prompts, tests); dt = clk() - t
    print(f"\n[validate] vLLM ASR on known triggers ({dt:.2f}s for {len(tests)*len(prompts)} gens):")
    for s, a in zip(tests, asrs): print(f"   {s:18} ASR={a}")

    fi, sl = slots[0]
    trig = torch.tensor([wordids[0], wordids[1]], device=model.device)

    # --- GCG GRADIENT STEP (the gadget): single trig ---
    for _ in range(2): grad_candidates(model, tok, fi, sl, trig, tgt, norm, 256)   # warmup
    t = clk()
    for _ in range(10): grad_candidates(model, tok, fi, sl, trig, tgt, norm, 256)
    print(f"\n[gcg] grad_candidates (1 trig, 1 prompt): {(clk()-t)/10*1000:.0f} ms/call")

    # --- HF target_nll over a pool of P candidates (batched forward) ---
    for P in (64, 256, 512):
        trials = trig.unsqueeze(0).repeat(P, 1).clone()
        t = clk(); _ = target_nll(model, tok, fi, sl, trials, tgt, norm); print(f"[hf-nll]  P={P:4d} (1 prompt): {(clk()-t)*1000:.0f} ms")

    # --- vLLM ASR over a pool of P candidates (batched, 4 prompts) ---
    for P in (64, 256, 512):
        cands = [f"{tok.decode([wordids[i]]).strip()} {tok.decode([wordids[i+1]]).strip()}" for i in range(P)]
        t = time.time(); _ = vllm_asr_batch(be, prompts, cands); print(f"[vllm-asr] P={P:4d} (4 prompts={P*4} gens): {time.time()-t:.2f}s")

    print("\n[done] benchmark complete", flush=True)

if __name__=='__main__':
    main()
