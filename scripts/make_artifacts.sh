#!/bin/bash
# Move the fold models of a finished final run into final/artifacts (for inference_from_artifacts.ipynb) and write
# SHA256SUMS. Run once, after the run's submission sha256 has been checked.
# usage: bash scripts/make_artifacts.sh final/work/run1
set -euo pipefail
cd /root/Dicefest
RUN=${1:?work dir of the finished run}
ART=final/artifacts
[ -d "$RUN/models" ] || { echo "no $RUN/models"; exit 1; }
[ -e "$ART/models" ] && { echo "$ART/models already exists, not overwriting"; exit 1; }
mkdir -p "$ART"
mv "$RUN/models" "$ART/models"
cp -r "$RUN/oof" "$ART/oof"
cp -r "$RUN/pred" "$ART/pred"
cp "$RUN/summary.json" "$ART/summary.json"
(cd "$ART" && find models oof pred summary.json -type f | LC_ALL=C sort | xargs sha256sum > SHA256SUMS)
echo "artifacts: $(du -sh $ART | cut -f1), $(wc -l < $ART/SHA256SUMS) files"
