# Step 6B Tuned Parameter Independent Validation

## Purpose

- Use the tuned parameters selected in Step 6A.
- Validate with independent seeds starting at seed=101.
- Use repeat=10 for the full validation run.
- Avoid reporting only the best result selected during the tuning search.

Raw input: `results/tuned_validation_raw.csv`

## Validation Results

| Instance | Family | Variant | Runs | BKS | Best Min | Best Mean | Gap Min % | Gap Mean % | Mean ms | Std ms | Parameters |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eil101 | QLSA | tuned | 10 | 629 | 632 | 638.600 | 0.477 | 1.526 | 421.933 | 31.535 | it=1000000, t0=1000, tf=0.001, chains=32, threads=8, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | SA | tuned | 10 | 629 | 632 | 639.800 | 0.477 | 1.717 | 190.869 | 9.162 | it=1000000, t0=100, tf=0.001, chains=32, threads=8 |
| eil76 | QLSA | tuned | 10 | 538 | 541 | 543.300 | 0.558 | 0.985 | 399.390 | 50.089 | it=1000000, t0=1000, tf=0.001, chains=32, threads=8, alpha=0.05, gamma=0.8, epsilon=0.2, policy=epsilon-greedy |
| eil76 | SA | tuned | 10 | 538 | 538 | 540.600 | 0.000 | 0.483 | 214.220 | 7.218 | it=1000000, t0=300, tf=0.001, chains=32, threads=8 |
| rat99 | QLSA | quality-first | 10 | 1211 | 1212 | 1215.500 | 0.083 | 0.372 | 854.307 | 93.014 | it=2000000, t0=1000, tf=0.001, chains=32, threads=8, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | tradeoff | 10 | 1211 | 1217 | 1224.000 | 0.495 | 1.073 | 379.518 | 38.895 | it=1000000, t0=1000, tf=0.0001, chains=32, threads=8, alpha=0.05, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | SA | tuned | 10 | 1211 | 1213 | 1221.600 | 0.165 | 0.875 | 206.171 | 6.208 | it=1000000, t0=3000, tf=0.001, chains=32, threads=8 |

## Comparison With Step 5B Default Parameters

### eil101
- QLSA tuned: default gap 1.272%, tuned validation min gap 0.477%, mean gap 1.526%.
- SA tuned: default gap 0.954%, tuned validation min gap 0.477%, mean gap 1.717%.

### eil76
- QLSA tuned: default gap 0.743%, tuned validation min gap 0.558%, mean gap 0.985%.
- SA tuned: default gap 0.186%, tuned validation min gap 0.000%, mean gap 0.483%.

### rat99
- QLSA quality-first: default gap 1.156%, tuned validation min gap 0.083%, mean gap 0.372%.
- QLSA tradeoff: default gap 1.156%, tuned validation min gap 0.495%, mean gap 1.073%.
- SA tuned: default gap 0.330%, tuned validation min gap 0.165%, mean gap 0.875%.

## Conclusions From Available Validation Rows

- eil101: best recorded validation config is QLSA tuned with min gap 0.477% and mean gap 1.526%.
- eil76: best recorded validation config is SA tuned with min gap 0.000% and mean gap 0.483%.
- rat99: best recorded validation config is QLSA quality-first with min gap 0.083% and mean gap 0.372%.
- rat99: QLSA still has a better minimum Gap than SA in this validation output.

## Notes

- Tuning improves solution quality relative to default parameters, but some configurations increase runtime.
- The final report should show both the default-parameter parallel speedup results and the tuned-parameter solution-quality results.
- Quick mode is only a script smoke test; full repeat=10 validation is required before making final claims.
