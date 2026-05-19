#!/usr/bin/env bash
# Build an ASAP7-routed RISC-V GDS via OpenROAD-flow-scripts (ORFS).
#
# This script documents the canonical ORFS commands for OpenLithoHub
# Phase 3. It runs ORFS *locally* — the same flow runs in CI via
# .github/workflows/build-asap7-mock-alu.yml on a Linux runner. Use
# the workflow when you're on macOS / Windows or don't want to install
# the toolchain locally; use this script when you have a Linux box and
# want faster iteration.
#
# Prerequisites:
#   - Linux (ORFS does not support macOS or Windows natively)
#   - Docker with the openroad/orfs image, OR a local ORFS install
#     with OPENROAD_EXE / YOSYS_EXE on PATH and the ORFS clone reachable
#     via the FLOW_HOME environment variable
#
# License: ASAP7 ships under BSD-3-Clause; ORFS itself under BSD-3-Clause.
# See DATA-LICENSES.md for OpenLithoHub's data-license policy.

set -euo pipefail

DESIGN="${1:-mock-alu}"
ORFS_REF="${ORFS_REF:-74b5f9610c107e9a155ac50e6eac4ad146e75344}"
ORFS_DIR="${ORFS_DIR:-/tmp/OpenROAD-flow-scripts}"
OUT_DIR="${OUT_DIR:-${PWD}/orfs-artifacts}"

usage() {
  cat >&2 <<EOF
Usage: $0 [design]

  design     ORFS design under flow/designs/asap7/ (default: mock-alu).
             Other small targets: riscv32i, riscv32i-mock-sram, ibex.

Environment:
  ORFS_REF   Git ref of OpenROAD-flow-scripts to use (default: pinned SHA).
  ORFS_DIR   Where to clone ORFS (default: /tmp/OpenROAD-flow-scripts).
  OUT_DIR    Where to copy the produced GDS (default: ./orfs-artifacts).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${ORFS_DIR}" ]]; then
  echo ">>> Cloning OpenROAD-flow-scripts into ${ORFS_DIR}"
  git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git "${ORFS_DIR}"
fi

(
  cd "${ORFS_DIR}"
  echo ">>> Checking out ORFS ${ORFS_REF}"
  git fetch origin
  git checkout "${ORFS_REF}"
  git submodule update --init --recursive
)

echo ">>> Running ORFS make for asap7/${DESIGN}"
(
  cd "${ORFS_DIR}/flow"
  if [[ -f ../env.sh ]]; then
    # shellcheck disable=SC1091
    . ../env.sh
  fi
  make DESIGN_CONFIG="./designs/asap7/${DESIGN}/config.mk"
)

mkdir -p "${OUT_DIR}"
GDS=$(find "${ORFS_DIR}/flow/results/asap7/${DESIGN}" -name '*.gds' | head -1)
if [[ -z "${GDS}" ]]; then
  echo "!!! No GDS produced under ${ORFS_DIR}/flow/results/asap7/${DESIGN}" >&2
  exit 1
fi
cp "${GDS}" "${OUT_DIR}/${DESIGN}.gds"
echo ">>> GDS saved to ${OUT_DIR}/${DESIGN}.gds"
echo ">>> Run benchmark: openlithohub eval run \\"
echo "      --dataset orfs --node 7nm --accept-license \\"
echo "      --data-root ${OUT_DIR}/${DESIGN}.gds \\"
echo "      --tile-nm 2000 --pixel-nm 1.0 --model dummy-identity"
