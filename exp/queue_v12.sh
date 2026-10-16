#!/bin/bash
# when p2_lepm and p2_gpm (short-schedule PLR MLPs with aggregates) exist: re-select members from everything
# available and build sub_v12 / sub_v12nc plus their group-residual-corrected versions
cd /root/Dicefest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
until [ -f exp/pred/p2_gpm.npy ] && [ -f exp/pred/p2_lepm.npy ]; do sleep 60; done
sleep 30
for v in v12:nn_,nn3_,nn4_,nn5_,n4_,n5_ v12nc:nn_,nn3_,nn4_,nn5_,n4_,n5_,c4_,cat_; do
  tag=${v%%:*}; ex=${v#*:}
  .venv/bin/python -u src/prune.py 25 --exclude $ex > exp/prune_$tag.log 2>&1
  G=$(grep "^gas runs:" exp/prune_$tag.log | cut -d' ' -f3); E=$(grep "^elec runs:" exp/prune_$tag.log | cut -d' ' -f3)
  .venv/bin/python -u src/make_submission.py $tag exp/pred/adopt_nb0.npy --gas $G --elec $E > exp/sub_$tag.log 2>&1
  .venv/bin/python -u src/resid_corr.py apply $tag ${tag}r --lam exp/resid_lam_g.json > exp/sub_${tag}r.log 2>&1
done
echo done >> exp/sub_v12ncr.log
