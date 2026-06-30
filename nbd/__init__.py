"""nbd — Neologism Backdoor Discovery gadget.

Reconstructs the trigger-discovery toolkit used by discover_tokens.py / grpo_bag_ngram.py:

  common         model ids, ground truth, prompt loading, chat-id building, the <neo> token.
  vllm_backend   frozen-base reward model (greedy ASR + target log-prob) on vLLM.
  stage1_repexp  the <neo>-based representation-exploration probes: embedding similarity and
                 logit-shift similarity (FUNCTIONAL neighbours of the trained neologism).
  rewards        small reward helpers (causal ablation).
"""
