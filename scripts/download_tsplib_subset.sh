#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"

instances=(
  ulysses16 gr17 ulysses22 gr24 bayg29 bays29 dantzig42 swiss42 gr48 hk48
  eil51 berlin52 st70 pr76 eil76 rat99 eil101
)

download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$out"
  else
    echo "Neither curl nor wget is available; cannot download TSPLIB files." >&2
    return 2
  fi
}

for name in "${instances[@]}"; do
  target="${DATA_DIR}/${name}.tsp"
  if [[ -f "${target}" ]]; then
    echo "[skip] ${name}.tsp already exists"
    continue
  fi

  tmp_gz="${DATA_DIR}/${name}.tsp.gz.tmp"
  tmp_tsp="${DATA_DIR}/${name}.tsp.tmp"
  gz_url="https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/${name}.tsp.gz"
  tsp_url="https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/${name}.tsp"

  if download_file "${gz_url}" "${tmp_gz}" 2>/dev/null; then
    if command -v gzip >/dev/null 2>&1; then
      gzip -dc "${tmp_gz}" > "${target}"
      rm -f "${tmp_gz}"
      echo "[ok] downloaded ${name}.tsp"
    else
      rm -f "${tmp_gz}"
      echo "[warn] downloaded gzip for ${name}, but gzip is unavailable; please download manually"
    fi
  elif download_file "${tsp_url}" "${tmp_tsp}" 2>/dev/null; then
    mv "${tmp_tsp}" "${target}"
    rm -f "${tmp_gz}"
    echo "[ok] downloaded ${name}.tsp"
  else
    rm -f "${tmp_gz}" "${tmp_tsp}"
    echo "[warn] could not download ${name}.tsp from the official TSPLIB URL; place it in data/ manually if needed"
  fi
done
