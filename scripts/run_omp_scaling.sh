#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
DATA_DIR="${ROOT_DIR}/data"
RESULTS_DIR="${ROOT_DIR}/results"
CSV_FILE="${RESULTS_DIR}/omp_scaling.csv"

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

input=""
iterations=1000000
if [[ -f "${DATA_DIR}/berlin52.tsp" ]]; then
  input="${DATA_DIR}/berlin52.tsp"
elif [[ -f "${DATA_DIR}/eil51.tsp" ]]; then
  input="${DATA_DIR}/eil51.tsp"
elif [[ -f "${ROOT_DIR}/tests/fixtures/square4.tsp" ]]; then
  input="${ROOT_DIR}/tests/fixtures/square4.tsp"
  iterations=10000
else
  echo "No input instance found. Expected data/berlin52.tsp, data/eil51.tsp, or tests/fixtures/square4.tsp." >&2
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

runtime_input="$(runtime_path "${input}")"

echo "algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves" > "${CSV_FILE}"

chains=8
repeat=3
seed=1
threads_list=(1 2 4 8)

echo "[info] input=${input}"
echo "[info] iterations=${iterations}"

for threads in "${threads_list[@]}"; do
  echo "[run] SA OMP threads=${threads}"
  "${TSP_SA}" \
    --input "${runtime_input}" \
    --parallel omp \
    --chains "${chains}" \
    --threads "${threads}" \
    --iterations "${iterations}" \
    --repeat "${repeat}" \
    --seed "${seed}" \
    --init nn \
    --csv-only >> "${CSV_FILE}"

  echo "[run] QLSA OMP threads=${threads}"
  "${TSP_SA}" \
    --input "${runtime_input}" \
    --qlsa \
    --parallel omp \
    --chains "${chains}" \
    --threads "${threads}" \
    --iterations "${iterations}" \
    --repeat "${repeat}" \
    --seed "${seed}" \
    --init nn \
    --alpha 0.1 \
    --gamma 0.9 \
    --epsilon 0.1 \
    --policy epsilon-greedy \
    --csv-only >> "${CSV_FILE}"
done

echo "OpenMP scaling CSV written to ${CSV_FILE}."
