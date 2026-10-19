#!/usr/bin/env bash
# Temporal grounding, adapted from the original run_scanrefer.sh.
set -euo pipefail
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --module scene_r1.training --task temporal "$@"
