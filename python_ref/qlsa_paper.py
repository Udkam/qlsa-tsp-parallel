#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style Q-Learning Assisted Simulated Annealing (QLSA).

Faithful to the reference paper's *candidate-leader* idea: at each iteration a
Q-learning policy selects which candidate leader guides the next 2-opt
Metropolis step. Candidate leaders are:

  0 = current solution
  1 = global best solution
  2 = random solution
  3 = double-bridge perturbed solution

This module also exposes ``run_engine`` so the State-Based variant (SB-QLSA)
can reuse the same loop with a diversity state.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, List

from tsplib_loader import (
    apply_2opt,
    delta_2opt,
    double_bridge,
    nearest_neighbor_tour,
    random_tour,
    tour_length,
)

NUM_ACTIONS = 4  # current, best, random, double-bridge


@dataclass
class RunResult:
    best_length: int
    final_length: int
    elapsed_ms: float
    accepted_moves: int
    improved_moves: int


def select_action(q_row: List[float], policy: str, epsilon: float,
                  softmax_temp: float, rng: random.Random) -> int:
    if policy == "softmax":
        m = max(q_row)
        weights = [math.exp((q - m) / max(softmax_temp, 1e-6)) for q in q_row]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for a, w in enumerate(weights):
            acc += w
            if r <= acc:
                return a
        return len(q_row) - 1
    # epsilon-greedy
    if rng.random() < epsilon:
        return rng.randrange(len(q_row))
    best_a, best_q = 0, q_row[0]
    for a in range(1, len(q_row)):
        if q_row[a] > best_q:
            best_q, best_a = q_row[a], a
    return best_a


def _one_2opt_metropolis(tour: List[int], length: int, dist: List[List[int]],
                         temperature: float, rng: random.Random) -> int:
    """Apply one random 2-opt move under Metropolis; return new length."""
    n = len(tour)
    i = 1 + rng.randrange(n - 1)
    k = i + rng.randrange(n - i)
    if k <= i:
        return length
    delta = delta_2opt(tour, dist, i, k)
    if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
        apply_2opt(tour, i, k)
        return length + delta
    return length


def run_engine(dist: List[List[int]], iterations: int, seed: int,
               policy: str = "epsilon-greedy", alpha: float = 0.1,
               gamma: float = 0.9, epsilon: float = 0.1, softmax_temp: float = 1.0,
               t0: float = 1000.0, tf: float = 1e-3, init: str = "nn",
               num_states: int = 1,
               state_fn: Callable[[List[int], List[int], int], int] | None = None) -> RunResult:
    rng = random.Random(seed)
    n = len(dist)
    if state_fn is None:
        state_fn = lambda cur, best, nn: 0  # noqa: E731 (stateless)

    current = nearest_neighbor_tour(dist, 0) if init == "nn" else random_tour(n, rng)
    current_len = tour_length(current, dist)
    best = current[:]
    best_len = current_len

    q_table = [[0.0] * NUM_ACTIONS for _ in range(num_states)]
    cooling = (tf / t0) ** (1.0 / max(1, iterations)) if t0 > 0 else 1.0
    temperature = t0
    accepted = 0
    improved = 0
    state = state_fn(current, best, n)

    start = time.perf_counter()
    for _ in range(iterations):
        action = select_action(q_table[state], policy, epsilon, softmax_temp, rng)

        # Build the leader implied by the chosen action.
        if action == 0:
            leader, leader_len = current[:], current_len
        elif action == 1:
            leader, leader_len = best[:], best_len
        elif action == 2:
            leader = random_tour(n, rng)
            leader_len = tour_length(leader, dist)
        else:
            leader = double_bridge(current, rng)
            leader_len = tour_length(leader, dist)

        # One 2-opt Metropolis refinement step on the leader.
        cand_len = _one_2opt_metropolis(leader, leader_len, dist, temperature, rng)

        # Move current to the candidate under Metropolis vs current.
        prev_len = current_len
        delta_total = cand_len - current_len
        if delta_total <= 0 or rng.random() < math.exp(-delta_total / max(temperature, 1e-12)):
            current, current_len = leader, cand_len
            accepted += 1
            if current_len < prev_len:
                improved += 1
            if current_len < best_len:
                best_len, best = current_len, current[:]

        reward = max(0.0, prev_len - current_len)
        next_state = state_fn(current, best, n)
        best_next = max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])
        state = next_state
        temperature *= cooling

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return RunResult(best_len, current_len, elapsed_ms, accepted, improved)


def run_qlsa(dist: List[List[int]], iterations: int, seed: int,
             policy: str = "epsilon-greedy", **kwargs) -> RunResult:
    """Stateless candidate-leader QLSA (single Q state)."""
    return run_engine(dist, iterations, seed, policy=policy, num_states=1,
                      state_fn=None, **kwargs)
