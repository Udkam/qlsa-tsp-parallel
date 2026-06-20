# Step 6A OpenMP Scaling Grid Analysis

## Purpose

This experiment measures the relationship between OpenMP threads and independent search chains.

## Summary Table

| Instance | Algorithm | Chains | Threads | Runs | Best | Gap % | Mean ms | Speedup | Efficiency % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| berlin52 | qlsa-omp | 32 | 1 | 3 | 7542 | 0.000 | 2072.596 | 1.000 | 100.00 |
| berlin52 | qlsa-omp | 32 | 2 | 3 | 7542 | 0.000 | 1085.162 | 1.910 | 95.50 |
| berlin52 | qlsa-omp | 32 | 4 | 3 | 7542 | 0.000 | 639.377 | 3.242 | 81.04 |
| berlin52 | qlsa-omp | 32 | 8 | 3 | 7542 | 0.000 | 373.035 | 5.556 | 69.45 |
| berlin52 | qlsa-omp | 32 | 12 | 3 | 7542 | 0.000 | 289.897 | 7.149 | 59.58 |
| berlin52 | qlsa-omp | 32 | 16 | 3 | 7542 | 0.000 | 260.653 | 7.952 | 49.70 |
| berlin52 | qlsa-omp | 64 | 1 | 3 | 7542 | 0.000 | 4389.891 | 1.000 | 100.00 |
| berlin52 | qlsa-omp | 64 | 2 | 3 | 7542 | 0.000 | 2147.987 | 2.044 | 102.19 |
| berlin52 | qlsa-omp | 64 | 4 | 3 | 7542 | 0.000 | 1155.080 | 3.801 | 95.01 |
| berlin52 | qlsa-omp | 64 | 8 | 3 | 7542 | 0.000 | 736.375 | 5.961 | 74.52 |
| berlin52 | qlsa-omp | 64 | 12 | 3 | 7542 | 0.000 | 619.371 | 7.088 | 59.06 |
| berlin52 | qlsa-omp | 64 | 16 | 3 | 7542 | 0.000 | 513.156 | 8.555 | 53.47 |
| berlin52 | sa-omp | 32 | 1 | 3 | 7542 | 0.000 | 1041.537 | 1.000 | 100.00 |
| berlin52 | sa-omp | 32 | 2 | 3 | 7542 | 0.000 | 502.941 | 2.071 | 103.54 |
| berlin52 | sa-omp | 32 | 4 | 3 | 7542 | 0.000 | 300.259 | 3.469 | 86.72 |
| berlin52 | sa-omp | 32 | 8 | 3 | 7542 | 0.000 | 176.544 | 5.900 | 73.74 |
| berlin52 | sa-omp | 32 | 12 | 3 | 7542 | 0.000 | 155.703 | 6.689 | 55.74 |
| berlin52 | sa-omp | 32 | 16 | 3 | 7542 | 0.000 | 130.269 | 7.995 | 49.97 |
| berlin52 | sa-omp | 64 | 1 | 3 | 7542 | 0.000 | 2204.808 | 1.000 | 100.00 |
| berlin52 | sa-omp | 64 | 2 | 3 | 7542 | 0.000 | 1048.832 | 2.102 | 105.11 |
| berlin52 | sa-omp | 64 | 4 | 3 | 7542 | 0.000 | 579.965 | 3.802 | 95.04 |
| berlin52 | sa-omp | 64 | 8 | 3 | 7542 | 0.000 | 351.141 | 6.279 | 78.49 |
| berlin52 | sa-omp | 64 | 12 | 3 | 7542 | 0.000 | 314.339 | 7.014 | 58.45 |
| berlin52 | sa-omp | 64 | 16 | 3 | 7542 | 0.000 | 302.680 | 7.284 | 45.53 |
| eil101 | qlsa-omp | 32 | 1 | 3 | 637 | 1.272 | 2233.288 | 1.000 | 100.00 |
| eil101 | qlsa-omp | 32 | 2 | 3 | 637 | 1.272 | 1251.098 | 1.785 | 89.25 |
| eil101 | qlsa-omp | 32 | 4 | 3 | 637 | 1.272 | 621.201 | 3.595 | 89.88 |
| eil101 | qlsa-omp | 32 | 8 | 3 | 637 | 1.272 | 353.132 | 6.324 | 79.05 |
| eil101 | qlsa-omp | 32 | 12 | 3 | 637 | 1.272 | 309.270 | 7.221 | 60.18 |
| eil101 | qlsa-omp | 32 | 16 | 3 | 637 | 1.272 | 273.429 | 8.168 | 51.05 |
| eil101 | qlsa-omp | 64 | 1 | 3 | 637 | 1.272 | 5469.115 | 1.000 | 100.00 |
| eil101 | qlsa-omp | 64 | 2 | 3 | 637 | 1.272 | 2410.658 | 2.269 | 113.44 |
| eil101 | qlsa-omp | 64 | 4 | 3 | 637 | 1.272 | 1232.611 | 4.437 | 110.93 |
| eil101 | qlsa-omp | 64 | 8 | 3 | 637 | 1.272 | 821.285 | 6.659 | 83.24 |
| eil101 | qlsa-omp | 64 | 12 | 3 | 637 | 1.272 | 619.148 | 8.833 | 73.61 |
| eil101 | qlsa-omp | 64 | 16 | 3 | 637 | 1.272 | 532.657 | 10.268 | 64.17 |
| eil101 | sa-omp | 32 | 1 | 3 | 635 | 0.954 | 1125.626 | 1.000 | 100.00 |
| eil101 | sa-omp | 32 | 2 | 3 | 635 | 0.954 | 590.003 | 1.908 | 95.39 |
| eil101 | sa-omp | 32 | 4 | 3 | 635 | 0.954 | 309.015 | 3.643 | 91.07 |
| eil101 | sa-omp | 32 | 8 | 3 | 635 | 0.954 | 186.191 | 6.046 | 75.57 |
| eil101 | sa-omp | 32 | 12 | 3 | 635 | 0.954 | 158.608 | 7.097 | 59.14 |
| eil101 | sa-omp | 32 | 16 | 3 | 635 | 0.954 | 143.742 | 7.831 | 48.94 |
| eil101 | sa-omp | 64 | 1 | 3 | 635 | 0.954 | 2280.312 | 1.000 | 100.00 |
| eil101 | sa-omp | 64 | 2 | 3 | 635 | 0.954 | 1140.369 | 2.000 | 99.98 |
| eil101 | sa-omp | 64 | 4 | 3 | 635 | 0.954 | 601.117 | 3.793 | 94.84 |
| eil101 | sa-omp | 64 | 8 | 3 | 635 | 0.954 | 371.130 | 6.144 | 76.80 |
| eil101 | sa-omp | 64 | 12 | 3 | 635 | 0.954 | 307.950 | 7.405 | 61.71 |
| eil101 | sa-omp | 64 | 16 | 3 | 635 | 0.954 | 286.831 | 7.950 | 49.69 |

## Best Combination

- Best score by quality then time: berlin52 sa-omp chains=32 threads=16 best=7542 gap=0.000% mean_ms=130.269.

## Notes

- Speedup is computed against the same instance, algorithm, and chains with threads=1.
- If the machine has fewer physical cores than requested threads, results still record the actual measured runtime.
- Full conclusions should use the full grid, not quick mode.
