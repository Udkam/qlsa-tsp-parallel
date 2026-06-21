#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style State-Based QLSA (SB-QLSA).

Extends candidate-leader QLSA with a *diversity state* derived from the Hamming
distance between the current tour and the global-best tour, giving a
state-action Q table. Two states are used:

  state 0 = low diversity  (current is close to best -> intensify)
  state 1 = high diversity (current is far from best -> explore)
"""

from __future__ import annotations

from typing import List

from qlsa_paper import RunResult, run_engine
from tsplib_loader import hamming_distance

NUM_STATES = 2
DIVERSITY_THRESHOLD = 0.5  # fraction of positions differing


def _diversity_state(current: List[int], best: List[int], n: int) -> int:
    if n == 0:
        return 0
    return 1 if (hamming_distance(current, best) / n) >= DIVERSITY_THRESHOLD else 0


def run_sb_qlsa(dist: List[List[int]], iterations: int, seed: int,
                policy: str = "epsilon-greedy", **kwargs) -> RunResult:
    return run_engine(dist, iterations, seed, policy=policy,
                      num_states=NUM_STATES, state_fn=_diversity_state, **kwargs)
