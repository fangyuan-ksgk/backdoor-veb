"""rewards.py — small reward helpers for the GRPO composer.

The default WINNING reward (fire * multiplicative minimality) lives in grpo_bag_ngram.py. This holds
the helper used by the optional `causal_len` reward: a causal-ablation of a composition (drop a word
and re-score), which contrasts the trigger's log-prob against a minimally perturbed counterfactual.
"""


def ablate(comp):
    """Drop the LAST word of a whitespace composition (causal counterfactual). '' if it had one word."""
    parts = comp.split()
    return " ".join(parts[:-1])
