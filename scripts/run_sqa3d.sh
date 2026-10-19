#!/usr/bin/env bash
# QA, adapted from the original run_sqa3d.sh.
set -euo pipefail
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --module scene_r1.training --task qa "$@"
