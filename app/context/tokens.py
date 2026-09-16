"""Token estimation for context budgeting (Slice 7A).

We do not call the LLM tokenizer API here. Instead we use a simple
character heuristic:

    estimated_tokens ≈ len(text) / CHARS_PER_TOKEN

Why a heuristic?

* **No extra dependency** — tiktoken is accurate but ties us to OpenAI
  tokenization; our local model (Ollama) may tokenize differently anyway.
* **Fast and deterministic** — budgeting runs on every ``ask()`` and agent
  round; it must be cheap.
* **Conservative enough** — we slightly over-estimate for code (lots of
  symbols and punctuation) so we are less likely to overflow the window.

Tradeoff: counts are approximate. Slice 7D may expose a tuning knob;
for learning, understanding *that* we budget is more important than
exact token parity with the model.
"""

from __future__ import annotations

import math

# Common rule of thumb for English prose; code often tokenizes slightly denser.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Return a non-negative token estimate for ``text``.

    Empty strings estimate to 0 tokens. Non-empty text always estimates
    to at least 1 token so tiny snippets still consume budget.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))
