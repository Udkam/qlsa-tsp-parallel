# TSPLIB95 Data

This directory is intentionally kept without bundled benchmark data.

You can either run:

```bash
bash scripts/download_tsplib_subset.sh
```

or manually download `.tsp` files from TSPLIB95-compatible mirrors and place them here, for example:

```text
data/berlin52.tsp
data/eil51.tsp
```

The baseline script skips missing files, so the project can build and test even when this directory contains no instances.
