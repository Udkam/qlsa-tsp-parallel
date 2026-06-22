# Step 6B Tuned Parameter Independent Validation

## Purpose

- Use the tuned parameters selected in Step 6A.
- Validate with independent seeds starting at seed=101.
- Use repeat=10 for the full validation run.
- Avoid reporting only the best result selected during the tuning search.

Raw input: `results/raw/final_hard_tuned_validation_raw.csv`

## Validation Results

| Instance | Family | Variant | Runs | BKS | Best Min | Best Mean | Gap Min % | Gap Mean % | Mean ms | Std ms | Parameters |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eil101 | QLSA | tuned | 10 | 629 | 630 | 637.100 | 0.159 | 1.288 | 351.473 | 31.984 | it=1000000, t0=1000, tf=0.001, chains=32, threads=8, alpha=0.1, gamma=0.8, epsilon=0.05, policy=epsilon-greedy |
| eil101 | SA | tuned | 10 | 629 | 634 | 638.300 | 0.795 | 1.479 | 190.972 | 7.006 | it=1000000, t0=100, tf=0.001, chains=32, threads=8 |
| eil76 | QLSA | tuned | 10 | 538 | 538 | 542.200 | 0.000 | 0.781 | 368.868 | 31.020 | it=1000000, t0=1000, tf=0.001, chains=32, threads=8, alpha=0.05, gamma=0.8, epsilon=0.2, policy=epsilon-greedy |
| eil76 | SA | tuned | 10 | 538 | 539 | 542.800 | 0.186 | 0.892 | 198.305 | 10.195 | it=1000000, t0=300, tf=0.001, chains=32, threads=8 |
| rat99 | QLSA | quality-first | 10 | 1211 | 1213 | 1217.200 | 0.165 | 0.512 | 737.504 | 93.530 | it=2000000, t0=1000, tf=0.001, chains=32, threads=8, alpha=0.2, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | QLSA | tradeoff | 10 | 1211 | 1214 | 1225.000 | 0.248 | 1.156 | 357.181 | 38.677 | it=1000000, t0=1000, tf=0.0001, chains=32, threads=8, alpha=0.05, gamma=0.95, epsilon=0.2, policy=epsilon-greedy |
| rat99 | SA | tuned | 10 | 1211 | 1216 | 1224.600 | 0.413 | 1.123 | 208.701 | 10.805 | it=1000000, t0=3000, tf=0.001, chains=32, threads=8 |

## Comparison With Step 5B Default Parameters

### eil101
- QLSA tuned: default gap 1.272%, tuned validation min gap 0.159%, mean gap 1.288%.
- SA tuned: default gap 0.954%, tuned validation min gap 0.795%, mean gap 1.479%.

### eil76
- QLSA tuned: default gap 0.743%, tuned validation min gap 0.000%, mean gap 0.781%.
- SA tuned: default gap 0.186%, tuned validation min gap 0.186%, mean gap 0.892%.

### rat99
- QLSA quality-first: default gap 1.156%, tuned validation min gap 0.165%, mean gap 0.512%.
- QLSA tradeoff: default gap 1.156%, tuned validation min gap 0.248%, mean gap 1.156%.
- SA tuned: default gap 0.330%, tuned validation min gap 0.413%, mean gap 1.123%.

## Conclusions From Available Validation Rows

- eil101: best recorded validation config is QLSA tuned with min gap 0.159% and mean gap 1.288%.
- eil76: best recorded validation config is QLSA tuned with min gap 0.000% and mean gap 0.781%.
- rat99: best recorded validation config is QLSA quality-first with min gap 0.165% and mean gap 0.512%.
- rat99: QLSA still has a better minimum Gap than SA in this validation output.

## Notes

- Tuning improves solution quality relative to default parameters, but some configurations increase runtime.
- The final report should show both the default-parameter parallel speedup results and the tuned-parameter solution-quality results.
- Quick mode is only a script smoke test; full repeat=10 validation is required before making final claims.
