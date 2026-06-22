# Step 6C Targeted Quality Experiment Analysis

## Purpose

- This stage is not a new full parameter search; it expands search budget around selected Step 6B configurations.
- Increasing chains launches more independent search chains in one experiment, which usually increases the chance of finding a better tour.
- Increasing iterations lets each chain search more deeply, but directly increases runtime.
- The final report should present both solution quality and runtime cost.
- The time-quality tradeoff score used here is empirical: score = gap_percent_min + 0.001 * elapsed_ms_mean.

Raw input: `results/raw/final_hard_targeted_raw.csv`

## Summary Table

| Instance | Family | Variant | Runs | Best | Gap Min % | Gap Mean % | Mean ms | Std ms | Score | Parameters |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| eil101 | QLSA | tuned-high-budget | 5 | 629 | 0.000 | 0.986 | 723.190 | 70.416 | 0.723 | it=1000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | QLSA | tuned-high-budget | 5 | 629 | 0.000 | 0.636 | 1537.426 | 133.686 | 1.537 | it=1000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | QLSA | tuned-high-budget | 5 | 630 | 0.159 | 0.541 | 1502.427 | 250.399 | 1.661 | it=2000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | QLSA | tuned-high-budget | 5 | 629 | 0.000 | 0.223 | 3097.566 | 326.952 | 3.098 | it=2000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | SA | tuned-high-budget | 5 | 632 | 0.477 | 1.335 | 381.229 | 4.249 | 0.858 | it=1000000, chains=64, threads=8, t0=100, tf=0.001 |
| eil101 | SA | tuned-high-budget | 5 | 631 | 0.318 | 0.795 | 729.832 | 16.837 | 1.048 | it=1000000, chains=128, threads=8, t0=100, tf=0.001 |
| eil101 | SA | tuned-high-budget | 5 | 630 | 0.159 | 0.731 | 736.261 | 23.246 | 0.895 | it=2000000, chains=64, threads=8, t0=100, tf=0.001 |
| eil101 | SA | tuned-high-budget | 5 | 629 | 0.000 | 0.477 | 1466.550 | 23.734 | 1.467 | it=2000000, chains=128, threads=8, t0=100, tf=0.001 |
| rat99 | QLSA | quality-first-high-budget | 5 | 1212 | 0.083 | 0.231 | 1506.279 | 114.146 | 1.589 | it=2000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | quality-first-high-budget | 5 | 1212 | 0.083 | 0.149 | 2930.032 | 402.217 | 3.013 | it=2000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | quality-first-high-budget | 5 | 1211 | 0.000 | 0.149 | 2273.361 | 353.493 | 2.273 | it=3000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | quality-first-high-budget | 5 | 1211 | 0.000 | 0.099 | 6807.419 | 4632.774 | 6.807 | it=3000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | SA | tuned-high-budget | 5 | 1216 | 0.413 | 0.677 | 405.264 | 4.610 | 0.818 | it=1000000, chains=64, threads=8, t0=3000, tf=0.001 |
| rat99 | SA | tuned-high-budget | 5 | 1216 | 0.413 | 0.562 | 837.705 | 39.076 | 1.251 | it=1000000, chains=128, threads=8, t0=3000, tf=0.001 |
| rat99 | SA | tuned-high-budget | 5 | 1213 | 0.165 | 0.380 | 795.143 | 19.401 | 0.960 | it=2000000, chains=64, threads=8, t0=3000, tf=0.001 |
| rat99 | SA | tuned-high-budget | 5 | 1212 | 0.083 | 0.248 | 1572.028 | 24.505 | 1.655 | it=2000000, chains=128, threads=8, t0=3000, tf=0.001 |

## Per-Instance Selection

### eil101
- Best quality: QLSA tuned-high-budget (it=2000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy), best=629, min gap=0.000%, mean gap=0.223%, reached BKS.
- Best time-quality tradeoff: QLSA tuned-high-budget (it=1000000, chains=64, threads=8, t0=1000, tf=0.001, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy), score=0.723, mean_ms=723.190.

### rat99
- Best quality: QLSA quality-first-high-budget (it=3000000, chains=128, threads=8, t0=1000, tf=0.001, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy), best=1211, min gap=0.000%, mean gap=0.099%, reached BKS.
- Best time-quality tradeoff: SA tuned-high-budget (it=1000000, chains=64, threads=8, t0=3000, tf=0.001), score=0.818, mean_ms=405.264.

## Comparison With Step 6B

- eil101 QLSA tuned it=1000000, chains=64: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.000%/0.986% (improved min, improved mean).
- eil101 QLSA tuned it=1000000, chains=128: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.000%/0.636% (improved min, improved mean).
- eil101 QLSA tuned it=2000000, chains=64: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.159%/0.541% (improved min, improved mean).
- eil101 QLSA tuned it=2000000, chains=128: Step 6B min/mean gap 0.477%/1.526%; Step 6C min/mean gap 0.000%/0.223% (improved min, improved mean).
- eil101 SA tuned it=1000000, chains=64: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.477%/1.335% (improved min, improved mean).
- eil101 SA tuned it=1000000, chains=128: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.318%/0.795% (improved min, improved mean).
- eil101 SA tuned it=2000000, chains=64: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.159%/0.731% (improved min, improved mean).
- eil101 SA tuned it=2000000, chains=128: Step 6B min/mean gap 0.477%/1.717%; Step 6C min/mean gap 0.000%/0.477% (improved min, improved mean).
- rat99 QLSA quality-first it=2000000, chains=64: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.083%/0.231% (improved min, improved mean).
- rat99 QLSA quality-first it=2000000, chains=128: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.083%/0.149% (improved min, improved mean).
- rat99 QLSA quality-first it=3000000, chains=64: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.000%/0.149% (improved min, improved mean).
- rat99 QLSA quality-first it=3000000, chains=128: Step 6B min/mean gap 0.083%/0.372%; Step 6C min/mean gap 0.000%/0.099% (improved min, improved mean).
- rat99 SA tuned it=1000000, chains=64: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.413%/0.677% (worse min, improved mean).
- rat99 SA tuned it=1000000, chains=128: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.413%/0.562% (worse min, improved mean).
- rat99 SA tuned it=2000000, chains=64: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.165%/0.380% (worse min, improved mean).
- rat99 SA tuned it=2000000, chains=128: Step 6B min/mean gap 0.165%/0.875%; Step 6C min/mean gap 0.083%/0.248% (improved min, improved mean).

## Notes

- Full conclusions require the complete repeat=5 targeted run; quick mode is only a smoke test.
- Because this stage increases budget, any quality improvement should be discussed together with elapsed time.
