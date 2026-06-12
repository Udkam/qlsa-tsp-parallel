# Final Experiment Summary

## 1. Experiment Groups

- Step 5B: default-parameter multi-instance speedup experiment. This is the main performance evidence for OpenMP multi-chain parallelization.
- Step 6B: tuned-parameter independent validation with seeds starting at 101. This is independent quality validation after parameter search.
- Step 6C: targeted high-budget quality experiment. This is supplemental evidence showing what happens when promising configurations receive more chains or iterations.

Final-report positioning:

- Main conclusions: Step 5B OpenMP speedup across multiple TSPLIB instances, plus Step 6B/6C quality improvements on harder instances.
- Supplementary analysis: CUDA results on berlin52 and targeted high-budget results, because CUDA is not the fastest path on the small instances tested and Step 6C spends more runtime to improve quality.

## 2. Default-Parameter OpenMP Speedup

| Instance | Family | Baseline Mean ms | OpenMP Mean ms | Speedup | Efficiency % | Best Length | Gap % |
|---|---|---:|---:|---:|---:|---:|---:|
| berlin52 | SA | 1043.053 | 202.402 | 5.153 | 64.417 | 7542 | 0.000 |
| berlin52 | QLSA | 2217.895 | 411.445 | 5.391 | 67.381 | 7542 | 0.000 |
| eil51 | SA | 1233.527 | 224.443 | 5.496 | 68.699 | 426 | 0.000 |
| eil51 | QLSA | 2371.958 | 428.979 | 5.529 | 69.116 | 426 | 0.000 |
| st70 | SA | 1243.066 | 247.472 | 5.023 | 62.788 | 675 | 0.000 |
| st70 | QLSA | 2356.846 | 560.222 | 4.207 | 52.587 | 675 | 0.000 |
| eil76 | SA | 1377.377 | 245.818 | 5.603 | 70.040 | 539 | 0.186 |
| eil76 | QLSA | 2439.045 | 490.690 | 4.971 | 62.133 | 542 | 0.743 |
| rat99 | SA | 1310.632 | 226.910 | 5.776 | 72.200 | 1215 | 0.330 |
| rat99 | QLSA | 2400.499 | 570.091 | 4.211 | 52.634 | 1225 | 1.156 |
| eil101 | SA | 1325.891 | 231.733 | 5.722 | 71.520 | 635 | 0.954 |
| eil101 | QLSA | 2490.072 | 445.503 | 5.589 | 69.867 | 637 | 1.272 |

Conclusions:

- SA OpenMP average speedup across the six Step 5B instances is about 5.46x.
- QLSA OpenMP average speedup across the six Step 5B instances is about 4.98x.
- OpenMP multi-chain parallelization is the primary performance improvement result for the final report.

## 3. Tuned-Parameter Independent Validation

| Instance | Family | Variant | Best Min | Min Gap % | Mean Gap % | Mean ms |
|---|---|---|---:|---:|---:|---:|
| eil101 | QLSA | tuned | 632 | 0.477 | 1.526 | 421.933 |
| eil101 | SA | tuned | 632 | 0.477 | 1.717 | 190.869 |
| eil76 | QLSA | tuned | 541 | 0.558 | 0.985 | 399.390 |
| eil76 | SA | tuned | 538 | 0.000 | 0.483 | 214.220 |
| rat99 | QLSA | quality-first | 1212 | 0.083 | 0.372 | 854.307 |
| rat99 | SA | tuned | 1213 | 0.165 | 0.875 | 206.171 |

Conclusions:

- eil76: SA tuned reached BKS.
- rat99: QLSA quality-first outperformed SA tuned in both min gap and mean gap.
- eil101: tuning improved min gap, but repeat=10 did not stably reproduce BKS.

## 4. Targeted High-Budget Quality Results

| Instance | Family | Variant | Best Min | Min Gap % | Mean Gap % | Mean ms |
|---|---|---|---:|---:|---:|---:|
| eil101 | QLSA | best-quality | 629 | 0.000 | 0.254 | 3348.545 |
| eil101 | QLSA | best-time-quality-tradeoff | 629 | 0.000 | 0.763 | 787.724 |
| eil101 | SA | best-quality | 629 | 0.000 | 0.445 | 1867.987 |
| rat99 | QLSA | best-quality | 1211 | 0.000 | 0.099 | 3424.631 |
| rat99 | SA | best-quality | 1212 | 0.083 | 0.330 | 1804.426 |

Conclusions:

- eil101: both SA and QLSA reached BKS after increasing budget.
- eil101 QLSA it=1e6 chains=64 reached BKS with relatively good time-quality tradeoff.
- rat99: QLSA it=2e6 chains=128 reached BKS=1211, while SA high-budget best remained 1212.
- rat99 demonstrates a clear case where QLSA improves solution quality over SA.

## 5. CUDA Result Positioning

- CUDA kernel has been successfully built and executed with Ninja/CUDA.
- On berlin52, CUDA reached BKS but was slower than OpenMP in the current implementation: SA CUDA mean 3540.677 ms vs SA OpenMP mean 196.097 ms; QLSA CUDA mean 8127.686 ms vs QLSA OpenMP mean 465.329 ms.
- CUDA is not the primary speedup result on small TSPLIB instances.
- The final report should present CUDA as an engineering extension and discuss limitations such as kernel launch overhead, memory access cost, and insufficient per-chain GPU work on small instances.

## 6. Recommended Final Claims

1. The project reproduced SA and implemented QLSA in C++.
2. The project implemented OpenMP multi-chain parallelization and achieved stable speedup across multiple TSPLIB instances.
3. OpenMP achieved about 5x average speedup without degrading solution quality.
4. Parameter tuning improved solution quality on harder instances.
5. QLSA reached BKS on rat99 under targeted high-budget configuration, while SA did not.
6. CUDA implementation was completed and validated, but not used as the primary speedup evidence for small instances.

## 7. Claims To Avoid

- Do not claim QLSA is always better than SA.
- Do not claim CUDA is faster than OpenMP.
- Do not claim all instances reach BKS under default parameters.
- Do not report tuning-search best results as independent validation conclusions.
- Do not overstate targeted high-budget results, because they increase runtime.
