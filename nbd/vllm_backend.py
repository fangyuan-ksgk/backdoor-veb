"""vllm_backend.py — the FROZEN base model as a fast reward / scoring engine.

Every reward in the pipeline (greedy ASR for "did the trigger fire", target log-prob for the dense
signal) is scored on the *frozen* organism via vLLM, never on a drifting policy. Inputs are raw
user-turn strings; this wraps each in the model's chat template before scoring.
"""
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from . import common as C


class VLLMBackend:
    def __init__(self, model_id, gpu_mem=0.5, max_len=1024, dtype="bfloat16"):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id, dtype=dtype, gpu_memory_utilization=gpu_mem,
                       max_model_len=max_len, enforce_eager=True, disable_log_stats=True)
        # target token ids (no special tokens) — what "I HATE YOU" tokenises to mid-stream.
        self.target_ids = self.tok.encode(C.TARGET, add_special_tokens=False)

    def _chat_ids(self, user_text):
        msgs = [{"role": "user", "content": user_text}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc   # transformers 5 -> BatchEncoding
        return list(ids)

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=8):
        """Greedy/sampled continuation of each prompt (wrapped as a user turn).
        Returns list[list[str]] (one inner list of `n` strings per prompt, input order)."""
        token_prompts = [{"prompt_token_ids": self._chat_ids(p)} for p in prompts]
        sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens)
        outs = self.llm.generate(token_prompts, sp, use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]

    def target_logprob(self, prompts):
        """Sum of per-token log-probs of TARGET ("I HATE YOU") immediately following each prompt.
        Computed via prompt_logprobs over the spliced prompt+target token sequence."""
        seqs, splits = [], []
        for p in prompts:
            cid = self._chat_ids(p)
            seqs.append({"prompt_token_ids": cid + self.target_ids})
            splits.append(len(cid))
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self.llm.generate(seqs, sp, use_tqdm=False)
        res = []
        for o, start, p in zip(outs, splits, prompts):
            pl = o.prompt_logprobs  # list aligned with prompt tokens; pl[0] is None
            full = self._chat_ids(p) + self.target_ids
            total = 0.0
            for pos in range(start, start + len(self.target_ids)):
                tid = full[pos]
                entry = pl[pos]
                total += entry[tid].logprob if entry and tid in entry else -20.0
            res.append(total)
        return res
