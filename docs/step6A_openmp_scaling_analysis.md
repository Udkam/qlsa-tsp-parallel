# Step 6A OpenMP Scaling Grid Analysis

## Purpose

This experiment measures the relationship between OpenMP threads and independent search chains.

## Summary Table

| Instance | Algorithm | Chains | Threads | Runs | Best | Gap % | Mean ms | Speedup | Efficiency % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| berlin52 | qlsa-omp | 8 | 1 | 1 | 7788 | 3.262 | 58.192 | 1.000 | 100.00 |
| berlin52 | qlsa-omp | 8 | 2 | 1 | 7788 | 3.262 | 32.530 | 1.789 | 89.44 |
| berlin52 | sa-omp | 8 | 1 | 1 | 7542 | 0.000 | 27.629 | 1.000 | 100.00 |
| berlin52 | sa-omp | 8 | 2 | 1 | 7542 | 0.000 | 15.865 | 1.742 | 87.08 |

## Best Combination

- Best score by quality then time: berlin52 sa-omp chains=8 threads=2 best=7542 gap=0.000% mean_ms=15.865.

## Notes

- Speedup is computed against the same instance, algorithm, and chains with threads=1.
- If the machine has fewer physical cores than requested threads, results still record the actual measured runtime.
- Full conclusions should use the full grid, not quick mode.
