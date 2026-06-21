#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style serial Simulated Annealing (SA) baseline.

2-opt neighborhood, Metropolis acceptance, exponential cooling. Written for
clarity and faithfulness to the reference paper, not for speed.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List

from tsplib_loader import (
    apply_2opt,
    delta_2opt,
    nearest_neighbor_tour,
    random_tour,
    tour_length,
)


@dataclass
class RunResult:
    best_length: int
    final_length: int
    elapsed_ms: float
    accepted_moves: int
    improved_moves: int


def run_sa(dist: List[List[int]], iterations: int, seed: int,
           t0: float = 1000.0, tf: float = 1e-3, init: str = "nn") -> RunResult:
    rng = random.Random(seed)
    n = len(dist)
    tour = nearest_neighbor_tour(dist, 0) if init == "nn" else random_tour(n, rng)
    current_len = tour_length(tour, dist)
    best_len = current_len
    best_tour = tour[:]

    # Geometric cooling: T_{t+1} = T_t * cooling, with cooling derived from t0/tf.
    cooling = (tf / t0) ** (1.0 / max(1, iterations)) if t0 > 0 else 1.0
    temperature = t0
    accepted = 0
    improved = 0

    start = time.perf_counter()
    for _ in range(iterations):
        i = 1 + rng.randrange(n - 1)
        k = i + rng.randrange(n - i)
        if k <= i:
            temperature *= cooling
            continue
        delta = delta_2opt(tour, dist, i, k)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            apply_2opt(tour, i, k)
            current_len += delta
            accepted += 1
            if delta < 0:
                improved += 1
            if current_len < best_len:
                best_len = current_len
                best_tour = tour[:]
        temperature *= cooling

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return RunResult(
        best_length=best_len,
        final_length=current_len,
        elapsed_ms=elapsed_ms,
        accepted_moves=accepted,
        improved_moves=improved,
    )
