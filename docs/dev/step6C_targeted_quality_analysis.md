# Step 6C Targeted Quality Experiment Analysis

## Purpose

- This stage is not a new full parameter search; it expands search budget around selected Step 6B configurations.
- Increasing chains launches more independent search chains in one experiment, which usually increases the chance of finding a better tour.
- Increasing iterations lets each chain search more deeply, but directly increases runtime.
- The final report should present both solution quality and runtime cost.
- The time-quality tradeoff score used here is empirical: score = gap_percent_min + 0.001 * elapsed_ms_mean.

Raw input: `results/targeted_quality_raw.csv`

## Summary Table

| Instance | Family | Variant | Runs | Best | Gap Min % | Gap Mean % | Mean ms | Std ms | Score | Parameters |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| eil101 | QLSA | tuned-high-budget | 5 | 629 | 0.000 | 0.763 | 787.724 | 17.315 | 0.788 | it=1000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | QLSA | tuned-high-budget | 5 | 629 | 0.000 | 0.572 | 1850.746 | 322.772 | 1.851 | it=1000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | QLSA | tuned-high-budget | 5 | 630 | 0.159 | 0.477 | 1857.507 | 244.379 | 2.016 | it=2000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | QLSA | tuned-high-budget | 5 | 629 | 0.000 | 0.254 | 3348.545 | 471.895 | 3.349 | it=2000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | SA | tuned-high-budget | 5 | 632 | 0.477 | 1.113 | 397.617 | 20.007 | 0.875 | it=1000000, chains=64, threads=8, t0=100, tf=0.001 |
| eil101 | SA | tuned-high-budget | 5 | 631 | 0.318 | 0.859 | 854.593 | 20.960 | 1.173 | it=1000000, chains=128, threads=8, t0=100, tf=0.001 |
| eil101 | SA | tuned-high-budget | 5 | 629 | 0.000 | 0.572 | 795.358 | 16.169 | 0.795 | it=2000000, chains=64, threads=8, t0=100, tf=0.001 |
| eil101 | SA | tuned-high-budget | 5 | 629 | 0.000 | 0.445 | 1867.987 | 360.656 | 1.868 | it=2000000, chains=128, threads=8, t0=100, tf=0.001 |
| rat99 | QLSA | quality-first-high-budget | 5 | 1213 | 0.165 | 0.165 | 1898.119 | 325.686 | 2.063 | it=2000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | quality-first-high-budget | 5 | 1211 | 0.000 | 0.099 | 3424.631 | 371.534 | 3.425 | it=2000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | quality-first-high-budget | 5 | 1212 | 0.083 | 0.215 | 2667.886 | 365.997 | 2.750 | it=3000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | quality-first-high-budget | 5 | 1211 | 0.000 | 0.116 | 5841.528 | 572.067 | 5.842 | it=3000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | SA | tuned-high-budget | 5 | 1216 | 0.413 | 0.809 | 445.850 | 22.519 | 0.859 | it=1000000, chains=64, threads=8, t0=3000, tf=0.001 |
| rat99 | SA | tuned-high-budget | 5 | 1215 | 0.330 | 0.578 | 894.463 | 20.127 | 1.225 | it=1000000, chains=128, threads=8, t0=3000, tf=0.001 |
| rat99 | SA | tuned-high-budget | 5 | 1212 | 0.083 | 0.446 | 889.806 | 33.682 | 0.972 | it=2000000, chains=64, threads=8, t0=3000, tf=0.001 |
| rat99 | SA | tuned-high-budget | 5 | 1212 | 0.083 | 0.330 | 1804.426 | 133.875 | 1.887 | it=2000000, chains=128, threads=8, t0=3000, tf=0.001 |

## Per-Instance Selection

### eil101
- Best quality: QLSA tuned-high-budget (it=2000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy), best=629, min gap=0.000%, mean gap=0.254%, reached BKS.
- Best time-quality tradeoff: QLSA tuned-high-budget (it=1000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy), score=0.788, mean_ms=787.724.

### rat99
- Best quality: QLSA quality-first-high-budget (it=2000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy), best=1211, min gap=0.000%, mean gap=0.099%, reached BKS.
- Best time-quality tradeoff: SA tuned-high-budget (it=1000000, chains=64, threads=8, t0=3000, tf=0.001), score=0.859, mean_ms=445.850.

## Comparison With Step 6B

- eil101 QLSA tuned it=1000000, chains=64: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.000%/0.763% (improved min, improved mean).
- eil101 QLSA tuned it=1000000, chains=128: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.000%/0.572% (improved min, improved mean).
- eil101 QLSA tuned it=2000000, chains=64: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.159%/0.477% (improved min, improved mean).
- eil101 QLSA tuned it=2000000, chains=128: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.000%/0.254% (improved min, improved mean).
- eil101 SA tuned it=1000000, chains=64: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.477%/1.113% (improved min, improved mean).
- eil101 SA tuned it=1000000, chains=128: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.318%/0.859% (improved min, improved mean).
- eil101 SA tuned it=2000000, chains=64: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.000%/0.572% (improved min, improved mean).
- eil101 SA tuned it=2000000, chains=128: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.000%/0.445% (improved min, improved mean).
- rat99 QLSA quality-first it=2000000, chains=64: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.165%/0.165% (worse min, improved mean).
- rat99 QLSA quality-first it=2000000, chains=128: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.000%/0.099% (improved min, improved mean).
- rat99 QLSA quality-first it=3000000, chains=64: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.083%/0.215% (improved min, improved mean).
- rat99 QLSA quality-first it=3000000, chains=128: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.000%/0.116% (improved min, improved mean).
- rat99 SA tuned it=1000000, chains=64: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.413%/0.809% (worse min, improved mean).
- rat99 SA tuned it=1000000, chains=128: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.330%/0.578% (worse min, improved mean).
- rat99 SA tuned it=2000000, chains=64: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.083%/0.446% (improved min, improved mean).
- rat99 SA tuned it=2000000, chains=128: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.083%/0.330% (improved min, improved mean).

## Notes

- The targeted repeat=5 run has been completed. The results can be used as targeted high-budget quality evidence, but should be interpreted together with runtime cost.
- Because this stage increases budget, any quality improvement should be discussed together with elapsed time.
