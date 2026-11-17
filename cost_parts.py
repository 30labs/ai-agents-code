"""Prices and recorded usage for chapter 14's arithmetic.

Prices are US dollars per million tokens, read from each
provider's pricing page on the date in the chapter's dated
table. A revision replaces the numbers and keeps the code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    "One model's list prices, per million tokens."

    input: float
    output: float
    write: float  # a prefix written to the cache
    read: float  # a prefix read back from the cache
    minimum: int  # shortest prompt the cache will hold


PRICES = {
    "sonnet-5": Price(2.00, 10.00, 2.50, 0.20, 1_024),
    "haiku-5.5": Price(0.10, 0.50, 0.125, 0.01, 512),
}

# Chapter 5's recorded run (fixtures/ch05-loop.json), one
# row per model call in order: prompt tokens, cached prompt
# tokens, completion tokens, and the response's `created`
# time in whole seconds. test_copies.py checks the copy.
CH05_LOOP = [
    (611, 16, 225, 1794945139),
    (678, 611, 230, 1794945144),
    (776, 678, 264, 1794945150),
    (861, 776, 494, 1794945160),
]
