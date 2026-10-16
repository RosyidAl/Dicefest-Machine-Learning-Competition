#!/bin/bash
# after full-data p1_epm_wM (PID 2624332): re-select members with it as a candidate, build sub_v10 / sub_v10nc
# and their group-residual-corrected versions sub_v10r / sub_v10ncr (weights exp/resid_lam.json)
cd /root/Dicefest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
while kill -0 2624332 2>/dev/null; do sleep 30; done
[ -f exp/pred/p1_epm_wM.npy ] || { echo "p1_epm_wM missing" > exp/sub_v10.log; exit 1; }
for v in v10:nn_,nn3_,nn4_,nn5_,n4_,n5_ v10nc:nn_,nn3_,nn4_,nn5_,n4_,n5_,c4_,cat_; do
  tag=${v%%:*}; ex=${v#*:}
  .venv/bin/python -u src/prune.py 25 --exclude $ex > exp/prune_$tag.log 2>&1
  G=$(grep "^gas runs:" exp/prune_$tag.log | cut -d' ' -f3); E=$(grep "^elec runs:" exp/prune_$tag.log | cut -d' ' -f3)
  .venv/bin/python -u src/make_submission.py $tag exp/pred/adopt_nb0.npy --gas $G --elec $E > exp/sub_$tag.log 2>&1
  .venv/bin/python -u src/resid_corr.py apply $tag ${tag}r > exp/sub_${tag}r.log 2>&1
done
echo done >> exp/sub_v10ncr.log
