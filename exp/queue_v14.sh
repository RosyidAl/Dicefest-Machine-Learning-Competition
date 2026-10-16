#!/bin/bash
# v14: everything from chains C (PLR formulations), D (t1_epm_wM), LGB (la_*), E (p3 elec PLR); xs_* (no subzone TE,
# meant for unseen-subzone rows only) are left out of the member selection
cd /root/Dicefest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
until [ -f exp/pred/p2_eui_wA.npy ] && [ -f exp/pred/t1_epm_wM.npy ] && [ -f exp/pred/la_gpgea.npy ] && [ -f exp/pred/p3_epm_wM.npy ]; do sleep 60; done
sleep 30
for v in v14:nn_,nn3_,nn4_,nn5_,n4_,n5_,xs_ v14nc:nn_,nn3_,nn4_,nn5_,n4_,n5_,xs_,c4_,cat_; do
  tag=${v%%:*}; ex=${v#*:}
  .venv/bin/python -u src/prune.py 25 --exclude $ex > exp/prune_$tag.log 2>&1
  G=$(grep "^gas runs:" exp/prune_$tag.log | cut -d' ' -f3); E=$(grep "^elec runs:" exp/prune_$tag.log | cut -d' ' -f3)
  .venv/bin/python -u src/make_submission.py $tag exp/pred/adopt_v2.npy --gas $G --elec $E > exp/sub_$tag.log 2>&1
  .venv/bin/python -u src/resid_corr.py apply $tag ${tag}r --lam exp/resid_lam_g.json > exp/sub_${tag}r.log 2>&1
done
echo done >> exp/sub_v14ncr.log
