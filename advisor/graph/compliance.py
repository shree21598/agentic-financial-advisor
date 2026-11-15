"""Input guardrails and output compliance post-processing."""
from __future__ import annotations

import re

DISCLAIMER = (
    "_This is general educational information, not personalized financial, tax, or legal advice. "
    "Consider consulting a licensed professional for your situation._"
)

_BLOCK = [
    (r"insider (trading|information|tip)", "trading on material non-public information"),
    (r"(evade|evasion|hide).{0,20}(tax|irs)", "evading taxes"),
    (r"launder", "money laundering"),
    (r"pump.{0,10}dump", "market manipulation"),
]


def check_input(query: str) -> str | None:
    """Return a refusal message if the query asks for clearly illegal activity."""
    for pat, what in _BLOCK:
        if re.search(pat, query, re.I):
            return (
                f"I can't help with {what}. I'm happy to help with legal strategies instead, such as "
                "tax-advantaged accounts, tax-loss harvesting, or building a diversified portfolio."
            )
    return None


_OVERPROMISE = re.compile(r"\b(guaranteed?|risk[- ]free|can'?t lose|sure thing)\b(?! (by|to be|rate))", re.I)


def finalize(answer: str) -> str:
    """Append the standard disclaimer (idempotent)."""
    if DISCLAIMER not in answer:
        answer = answer.rstrip() + "\n\n" + DISCLAIMER
    return answer


def overpromise_flags(answer: str) -> list[str]:
    return [m.group(0) for m in _OVERPROMISE.finditer(answer)]
