#!/usr/bin/env bash
# mem2mem footprint ladder after the bs8/clip128 OOM: fine bs at clip 128, plus clip 96/64 references.
set -euo pipefail
python -u experiments/memmaze-dynamics/bs_search.py --arms mem2mem --bs 1 2 4 6 --clip-len 128
python -u experiments/memmaze-dynamics/bs_search.py --arms mem2mem --bs 4 6 8 --clip-len 96
python -u experiments/memmaze-dynamics/bs_search.py --arms mem2mem --bs 6 8 12 --clip-len 64
echo "CALIB2 DONE"
