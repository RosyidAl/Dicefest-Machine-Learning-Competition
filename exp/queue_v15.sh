#!/bin/bash
# v15: v14 candidates + the multi-task NN members m1_gpm / m1_epm; then adoption v2, group residual correction and the
# train-duplicate copy (dup_fix) -> sub_v15rd / sub_v15ncrd
cd /root/Dicefest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for v in v15:nn_,nn3_,nn4_,nn5_,n4_,n5_,xs_,smk_ v15nc:nn_,nn3_,nn4_,nn5_,n4_,n5_,xs_,smk_,c4_,cat_; do
  tag=${v%%:*}; ex=${v#*:}
  .venv/bin/python -u src/prune.py 25 --exclude $ex > exp/prune_$tag.log 2>&1
  G=$(grep "^gas runs:" exp/prune_$tag.log | cut -d' ' -f3); E=$(grep "^elec runs:" exp/prune_$tag.log | cut -d' ' -f3)
  .venv/bin/python -u src/make_submission.py $tag exp/pred/adopt_v2.npy --gas $G --elec $E > exp/sub_$tag.log 2>&1
  .venv/bin/python -u src/resid_corr.py apply $tag ${tag}r --lam exp/resid_lam_g.json > exp/sub_${tag}r.log 2>&1
  .venv/bin/python -u src/dup_fix.py ${tag}r ${tag}rd >> exp/sub_${tag}r.log 2>&1
done
echo done >> exp/sub_v15ncr.log
