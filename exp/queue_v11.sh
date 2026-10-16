#!/bin/bash
# after full-data p1_gpm (PID 2631651) and the v10 build (PID 2628782): re-select members, build sub_v11 / sub_v11nc
# and their corrected versions with weights fitted on a holdout blend that includes p1_gpm (exp/resid_lam_g.json)
cd /root/Dicefest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
while kill -0 2631651 2>/dev/null || kill -0 2628782 2>/dev/null; do sleep 30; done
[ -f exp/pred/p1_gpm.npy ] || { echo "p1_gpm missing" > exp/sub_v11.log; exit 1; }
for v in v11:nn_,nn3_,nn4_,nn5_,n4_,n5_ v11nc:nn_,nn3_,nn4_,nn5_,n4_,n5_,c4_,cat_; do
  tag=${v%%:*}; ex=${v#*:}
  .venv/bin/python -u src/prune.py 25 --exclude $ex > exp/prune_$tag.log 2>&1
  G=$(grep "^gas runs:" exp/prune_$tag.log | cut -d' ' -f3); E=$(grep "^elec runs:" exp/prune_$tag.log | cut -d' ' -f3)
  .venv/bin/python -u src/make_submission.py $tag exp/pred/adopt_nb0.npy --gas $G --elec $E > exp/sub_$tag.log 2>&1
  .venv/bin/python -u src/resid_corr.py apply $tag ${tag}r --lam exp/resid_lam_g.json > exp/sub_${tag}r.log 2>&1
done
echo done >> exp/sub_v11ncr.log
