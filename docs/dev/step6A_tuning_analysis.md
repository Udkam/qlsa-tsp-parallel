# Step 6A Parameter Tuning Analysis

## Purpose

This stage searches SA and QLSA parameters to reduce the remaining Gap on eil76, rat99, and eil101, while keeping the algorithm implementation unchanged.

## Search Space

- SA: iterations in {1e6, 2e6}, t0 in {100, 300, 1000, 3000}, tf in {1e-3, 1e-4, 1e-5}.
- QLSA stage 1: t0=1000, tf=1e-3, iterations=1e6, alpha/gamma/epsilon/policy grid.
- QLSA stage 2: top stage-1 configurations per instance are expanded over t0/tf/iterations.
- Quick mode uses a reduced eil76-only grid for smoke testing.
- Tradeoff score is empirical: score = gap_percent + 0.001 * elapsed_ms_mean.

Raw input: `results/tuning_raw.csv`

## Best Configurations

| Instance | Family | Best Quality Gap % | Best Quality Config | Tradeoff Gap % | Tradeoff Mean ms | Tradeoff Config |
|---|---|---:|---|---:|---:|---|
| eil101 | SA | 0.000 | it=1000000, t0=100, tf=0.001 | 0.000 | 180.150 | it=1000000, t0=100, tf=0.001 |
| eil101 | QLSA | 0.000 | it=1000000, t0=1000, tf=0.001, a=0.1, g=0.8, eps=0.05, epsilon-greedy | 0.000 | 368.345 | it=1000000, t0=1000, tf=0.001, a=0.1, g=0.8, eps=0.05, epsilon-greedy |
| eil76 | SA | 0.000 | it=1000000, t0=300, tf=0.001 | 0.000 | 183.370 | it=1000000, t0=300, tf=0.001 |
| eil76 | QLSA | 0.000 | it=1000000, t0=1000, tf=0.001, a=0.05, g=0.8, eps=0.2, epsilon-greedy | 0.000 | 345.918 | it=1000000, t0=1000, tf=0.001, a=0.05, g=0.8, eps=0.2, epsilon-greedy |
| rat99 | SA | 0.083 | it=1000000, t0=3000, tf=0.001 | 0.083 | 191.179 | it=1000000, t0=3000, tf=0.001 |
| rat99 | QLSA | 0.000 | it=2000000, t0=1000, tf=0.001, a=0.2, g=0.95, eps=0.2, epsilon-greedy | 0.083 | 355.616 | it=1000000, t0=1000, tf=0.0001, a=0.05, g=0.95, eps=0.2, epsilon-greedy |

## Comparison With Step 5B Defaults

### eil101
- SA: previous gap 0.954%, tuned best gap 0.000%, previous best 635, tuned best 629.
- QLSA: previous gap 1.272%, tuned best gap 0.000%, previous best 637, tuned best 629.

### eil76
- SA: previous gap 0.186%, tuned best gap 0.000%, previous best 539, tuned best 538.
- QLSA: previous gap 0.743%, tuned best gap 0.000%, previous best 542, tuned best 538.

### rat99
- SA: previous gap 0.330%, tuned best gap 0.083%, previous best 1215, tuned best 1212.
- QLSA: previous gap 1.156%, tuned best gap 0.000%, previous best 1225, tuned best 1211.

## QLSA vs SA Observation

- eil101: QLSA matched SA quality in this tuning run.
- eil76: QLSA matched SA quality in this tuning run.
- rat99: QLSA found a lower Gap than SA in this tuning run.

## Next Steps

- Run the full tuning grid before drawing final conclusions.
- Promote the best SA/QLSA configurations into the final multi-instance experiment matrix.
- Keep CUDA conclusions separate unless the CUDA runs are confirmed non-fallback and competitive on larger instances.
