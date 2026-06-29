#!/usr/bin/env bash
# One-shot Memory-Maze 9x9 tokenizer prep on the cluster (single H100 allocation):
#   [1/3] batch-size search on the H100 (LOCKED config) -> recommended --batch-size (streamed early)
#   [2/3] download a train shard from Google Drive via gdown
#   [3/3] convert the unzipped .npz trajectories -> one mmappable data/memmaze9x9.npy
#
# bs-search runs first and is NON-fatal, so its result lands in the log even if the (longer) download
# later fails/throttles. Usage (via submit_job.sh -- bash experiments/memmaze-tokenizer/cluster_prep.sh [PARTS]):
#   default PARTS = train-part0 (~10% of the 29k-traj train set). Pass e.g. "train-part0 train-part1".
set -uo pipefail
PARTS="${1:-train-part0}"

echo "########## [1/3] batch-size search (LOCKED config, LPIPS on) ##########"
python -u experiments/memmaze-tokenizer/bs_search.py --lpips || echo "WARN: bs_search failed; continuing to data prep"

echo "########## [2/3] download parts: $PARTS ##########"
pip install --quiet gdown
set -e
python -u experiments/memmaze-tokenizer/download_memmaze.py --parts $PARTS --out-dir data/memmaze9x9_raw --unzip

echo "########## [3/3] convert -> data/memmaze9x9.npy ##########"
python -u experiments/memmaze-tokenizer/convert_memmaze.py --raw data/memmaze9x9_raw --out data/memmaze9x9.npy
echo "########## PREP DONE ##########"
