#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
DATA_DIR="${ROOT_DIR}/data"
RESULTS_DIR="${ROOT_DIR}/results"
CSV_FILE="${RESULTS_DIR}/baseline_sa.csv"

instances=(
  ulysses16 gr17 ulysses22 gr24 bayg29 bays29 dantzig42 swiss42 gr48 hk48
  eil51 berlin52 st70 pr76 eil76 rat99 eil101
)

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

if command -v cmake >/dev/null 2>&1; then
  CMAKE_BIN="cmake"
elif command -v cmake.exe >/dev/null 2>&1; then
  CMAKE_BIN="cmake.exe"
else
  echo "Could not find cmake or cmake.exe in PATH." >&2
  exit 1
fi

cmake_source="${ROOT_DIR}"
cmake_build="${BUILD_DIR}"
if [[ "${CMAKE_BIN}" == "cmake.exe" ]] && command -v wslpath >/dev/null 2>&1; then
  cmake_source="$(wslpath -w "${ROOT_DIR}")"
  cmake_build="$(wslpath -w "${BUILD_DIR}")"
fi

"${CMAKE_BIN}" -S "${cmake_source}" -B "${cmake_build}" -DCMAKE_BUILD_TYPE=Release
"${CMAKE_BIN}" --build "${cmake_build}" --config Release --parallel

exe_candidates=(
  "${BUILD_DIR}/tsp_sa"
  "${BUILD_DIR}/tsp_sa.exe"
  "${BUILD_DIR}/Release/tsp_sa"
  "${BUILD_DIR}/Release/tsp_sa.exe"
)

TSP_SA=""
for candidate in "${exe_candidates[@]}"; do
  if [[ -x "${candidate}" || -f "${candidate}" ]]; then
    TSP_SA="${candidate}"
    break
  fi
done

if [[ -z "${TSP_SA}" ]]; then
  echo "Could not find tsp_sa executable after build." >&2
  exit 1
fi

runtime_path() {
  local path="$1"
  if [[ "${TSP_SA}" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
    wslpath -w "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

echo "algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves" > "${CSV_FILE}"

ran_any=0
for name in "${instances[@]}"; do
  input="${DATA_DIR}/${name}.tsp"
  if [[ ! -f "${input}" ]]; then
    echo "[skip] ${name}.tsp not found in data/"
    continue
  fi

  ran_any=1
  echo "[run] ${name}"
  runtime_input="$(runtime_path "${input}")"
  "${TSP_SA}" \
    --input "${runtime_input}" \
    --iterations 1000000 \
    --repeat 3 \
    --seed 1 \
    --init nn \
    --csv-only >> "${CSV_FILE}"
done

if [[ "${ran_any}" -eq 0 ]]; then
  echo "No TSPLIB .tsp files were found. CSV header was still written to ${CSV_FILE}."
else
  echo "Baseline CSV written to ${CSV_FILE}."
fi
