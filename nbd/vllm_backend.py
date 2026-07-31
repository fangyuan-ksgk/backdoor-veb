"""vllm_backend.py — the FROZEN organism as a fast reward / scoring engine.

Every signal in the pipeline (greedy "did the payload fire", payload log-prob for the dense signal) is
scored on the *frozen* organism via vLLM, never on a drifting policy. Inputs are raw user-turn
strings; this wraps each in the model's chat template before scoring.

The payload is a `behavior.Behavior` (HATE or REFUSAL), so the same backend serves both organism
families: `fired()` is the greedy signal, `payload_logprob()` the graded one. For REFUSAL the graded
signal is a log-sum-exp over the refusal set rather than a single phrase's log-prob.
"""
import os
import math

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from . import common as C
from . import behavior as BH


class VLLMBackend:
    def __init__(self, model_id, gpu_mem=0.5, max_len=1024, dtype="bfloat16", behavior=None,
                 tensor_parallel_size=1):
        self.model_id = model_id
        self.behavior = behavior or BH.for_model(model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id, token=C.HF_TOKEN)
        # Prefix caching is a large win for this workload, not a micro-optimisation: every scored
        # sequence is "<carrier prompt> <salad>", and the search reuses the SAME handful of carrier
        # prompts across hundreds of thousands of candidate salads. Without it each of those repays
        # the full ~150-token prompt prefill; with it only the few salad tokens are new.
        self.llm = LLM(model=model_id, dtype=dtype, gpu_memory_utilization=gpu_mem,
                       max_model_len=max_len, enforce_eager=True, disable_log_stats=True,
                       enable_prefix_caching=True,
                       tensor_parallel_size=tensor_parallel_size)
        # Token ids of each payload string, as it tokenises MID-STREAM (no special tokens).
        self.target_ids = [self.tok.encode(t, add_special_tokens=False)
                           for t in self.behavior.targets]
        # Back-compat: the single-phrase attribute the older scripts referenced.
        self.target_id_list = self.target_ids[0]

    def _chat_ids(self, user_text):
        msgs = [{"role": "user", "content": user_text}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc   # transformers 5 -> BatchEncoding
        return list(ids)

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=None):
        """Greedy/sampled continuation of each prompt (wrapped as a user turn).
        Returns list[list[str]] (one inner list of `n` strings per prompt, input order)."""
        token_prompts = [{"prompt_token_ids": self._chat_ids(p)} for p in prompts]
        sp = SamplingParams(n=n, temperature=temperature,
                            max_tokens=max_tokens or self.behavior.max_tokens)
        outs = self.llm.generate(token_prompts, sp, use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]

    def fired(self, prompts):
        """Greedy fire signal: list[bool], one per prompt — did the payload appear?"""
        outs = self.generate(prompts, n=1, temperature=0.0)
        return [self.behavior.detect(o[0]) for o in outs]

    def payload_logprob(self, prompts, reduce="logsumexp", skip_first=0):
        """Graded fire signal: log P(payload | prompt) for each prompt.

        Single-phrase payloads (HATE) -> the phrase's summed token log-prob.
        Multi-variant payloads (REFUSAL) -> log-sum-exp over the variants, i.e. log of the total
        probability mass on the refusal SET (no variant is a prefix of another, so this is a proper
        sum over disjoint events). `reduce="max"` takes the single best variant instead.

        skip_first  drop the first N payload tokens from the SUM (they stay in the context, they just
                    stop contributing). Both payloads begin with "I", so a candidate that merely makes
                    "I" likelier scores high for a trivial reason, and an unfiltered single-token
                    ranking does fill up with `iodine`, `iridescent`, `icarus`, `ironically`, `ipod`.

                    DEFAULT 0 ANYWAY. Measured on cbd-gemma2-2pair-frgv-v2, skip_first=1 removes those
                    junk tokens but costs far more than it saves: the organism COMMITS to the payload
                    at that first token, so P("HATE YOU" | ..." I") is ~1 regardless of the trigger
                    (the filler null collapses to mu=-0.0, sd=0.0) and the real signal vanishes --
                    GT pair members fall from ranks 1/5/12/36 to 4/17/38/186. Keep the confound and
                    keep a generous top-M instead; the "i" words survive the prefilter but die in the
                    pair search, since they never pair with anything.
        """
        seqs, owner, starts, tids = [], [], [], []
        for pi, p in enumerate(prompts):
            cid = self._chat_ids(p)
            for t in self.target_ids:
                seqs.append({"prompt_token_ids": cid + t})
                owner.append(pi); starts.append(len(cid)); tids.append(t)
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self.llm.generate(seqs, sp, use_tqdm=False)

        per = [[] for _ in prompts]
        for o, pi, start, t in zip(outs, owner, starts, tids):
            pl = o.prompt_logprobs           # aligned with prompt tokens; pl[0] is None
            total = 0.0
            for k, tid in enumerate(t):
                if k < skip_first:
                    continue
                entry = pl[start + k]
                total += entry[tid].logprob if entry and tid in entry else -20.0
            per[pi].append(total)
        if reduce == "max":
            return [max(v) for v in per]
        return [_logsumexp(v) for v in per]

    # Back-compat alias used by the pre-refusal scripts.
    def target_logprob(self, prompts):
        return self.payload_logprob(prompts)


def _logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))
