#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TSPLIB95 loader and shared tour utilities for the Python faithful baseline.

This module mirrors the C++ side closely enough that best-tour lengths are
directly comparable: integer distances use the TSPLIB ``nint`` convention
(``int(x + 0.5)``). It is written for clarity, not speed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Instance:
    name: str
    dimension: int
    distances: List[List[int]]  # n x n integer distance matrix


def _nint(value: float) -> int:
    """TSPLIB nearest-integer convention."""
    return int(value + 0.5)


def _euc_2d(a, b) -> int:
    return _nint(math.hypot(a[0] - b[0], a[1] - b[1]))


def _ceil_2d(a, b) -> int:
    return int(math.ceil(math.hypot(a[0] - b[0], a[1] - b[1])))


def _att(a, b) -> int:
    rij = math.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 10.0)
    tij = _nint(rij)
    return tij + 1 if tij < rij else tij


def _geo_radians(coord: float) -> float:
    deg = int(coord)
    minutes = coord - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def _geo(a, b) -> int:
    rrr = 6378.388
    lat_a, lon_a = _geo_radians(a[0]), _geo_radians(a[1])
    lat_b, lon_b = _geo_radians(b[0]), _geo_radians(b[1])
    q1 = math.cos(lon_a - lon_b)
    q2 = math.cos(lat_a - lat_b)
    q3 = math.cos(lat_a + lat_b)
    return int(rrr * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)


_COORD_FUNCS = {
    "EUC_2D": _euc_2d,
    "CEIL_2D": _ceil_2d,
    "ATT": _att,
    "GEO": _geo,
}


def load_instance(path: str | Path) -> Instance:
    """Parse a TSPLIB95 ``.tsp`` file into an integer distance matrix.

    Supports coordinate types EUC_2D / CEIL_2D / ATT / GEO and EXPLICIT
    FULL_MATRIX, which covers every instance used by this project.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    name = path.stem
    dimension = 0
    edge_type = "EUC_2D"
    edge_format = "FUNCTION"
    section = None
    coords: List[tuple] = []
    explicit_values: List[float] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NAME"):
            name = line.split(":", 1)[1].strip() or name
            continue
        if upper.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1].strip())
            continue
        if upper.startswith("EDGE_WEIGHT_TYPE"):
            edge_type = line.split(":", 1)[1].strip().upper()
            continue
        if upper.startswith("EDGE_WEIGHT_FORMAT"):
            edge_format = line.split(":", 1)[1].strip().upper()
            continue
        if upper.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        if upper.startswith("EDGE_WEIGHT_SECTION"):
            section = "weights"
            continue
        if upper.startswith("EOF") or upper.startswith("DISPLAY_DATA"):
            section = None
            continue
        if section == "coords":
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2])))
        elif section == "weights":
            explicit_values.extend(float(v) for v in line.split())

    if dimension == 0:
        raise ValueError(f"{path}: missing DIMENSION")

    if edge_type == "EXPLICIT":
        distances = _build_explicit(dimension, edge_format, explicit_values)
    else:
        func = _COORD_FUNCS.get(edge_type)
        if func is None:
            raise ValueError(f"{path}: unsupported EDGE_WEIGHT_TYPE {edge_type}")
        distances = [[0] * dimension for _ in range(dimension)]
        for i in range(dimension):
            for j in range(i + 1, dimension):
                d = func(coords[i], coords[j])
                distances[i][j] = d
                distances[j][i] = d

    return Instance(name=name, dimension=dimension, distances=distances)


def _build_explicit(n: int, fmt: str, values: List[float]) -> List[List[int]]:
    dist = [[0] * n for _ in range(n)]
    it = iter(values)
    if fmt == "FULL_MATRIX":
        for i in range(n):
            for j in range(n):
                dist[i][j] = int(next(it))
    elif fmt in ("UPPER_ROW", "UPPER_DIAG_ROW"):
        for i in range(n):
            start = i if fmt == "UPPER_DIAG_ROW" else i + 1
            for j in range(start, n):
                v = int(next(it))
                dist[i][j] = v
                dist[j][i] = v
    elif fmt in ("LOWER_ROW", "LOWER_DIAG_ROW"):
        for i in range(n):
            end = i + 1 if fmt == "LOWER_DIAG_ROW" else i
            for j in range(end):
                v = int(next(it))
                dist[i][j] = v
                dist[j][i] = v
    else:
        raise ValueError(f"unsupported EDGE_WEIGHT_FORMAT {fmt}")
    return dist


# --------------------------------------------------------------------------
# Shared tour utilities (used by sa_paper / qlsa_paper / sb_qlsa_paper).
# --------------------------------------------------------------------------

def tour_length(tour: List[int], dist: List[List[int]]) -> int:
    n = len(tour)
    total = 0
    for i in range(n):
        total += dist[tour[i]][tour[(i + 1) % n]]
    return total


def nearest_neighbor_tour(dist: List[List[int]], start: int = 0) -> List[int]:
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        best_j, best_d = -1, None
        row = dist[current]
        for j in range(n):
            if not visited[j] and (best_d is None or row[j] < best_d):
                best_d, best_j = row[j], j
        visited[best_j] = True
        tour.append(best_j)
        current = best_j
    return tour


def random_tour(n: int, rng: random.Random) -> List[int]:
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def delta_2opt(tour: List[int], dist: List[List[int]], i: int, k: int) -> int:
    """Length change of reversing tour[i..k] (1 <= i < k <= n-1)."""
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[k], tour[(k + 1) % n]
    return dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]


def apply_2opt(tour: List[int], i: int, k: int) -> None:
    tour[i:k + 1] = reversed(tour[i:k + 1])


def double_bridge(tour: List[int], rng: random.Random) -> List[int]:
    """Classic 4-opt double-bridge perturbation; returns a valid permutation."""
    n = len(tour)
    if n < 8:
        out = tour[:]
        # small instances: fall back to a single random 2-opt-like swap
        i, j = rng.randrange(n), rng.randrange(n)
        out[i], out[j] = out[j], out[i]
        return out
    p1 = 1 + rng.randrange(n - 3)
    p2 = p1 + 1 + rng.randrange(n - p1 - 2)
    p3 = p2 + 1 + rng.randrange(n - p2 - 1)
    return tour[:p1] + tour[p2:p3] + tour[p1:p2] + tour[p3:]


def hamming_distance(a: List[int], b: List[int]) -> int:
    """Position-wise Hamming distance between two tours of equal length."""
    return sum(1 for x, y in zip(a, b) if x != y)
