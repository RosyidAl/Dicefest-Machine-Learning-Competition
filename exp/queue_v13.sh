#!/bin/bash
# chain C (PLR formulations) after chain B, chain D (TabM) after chain A, then build v13 once t1_gpm and
# p2_leui_wA exist
cd /root/Dicefest
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(while kill -0 2708287 2>/dev/null; do sleep 60; done
 .venv/bin/python -u src/exp_v11_nn.py p2_gpn,p2_gas,p2_elec,p2_epa,p2_eui_wA > exp/v11_nn_c.log 2>&1) &
(while kill -0 2698027 2>/dev/null; do sleep 60; done
 .venv/bin/python -u src/exp_v11_nn.py t1_epm_wM,t1_gpgea > exp/v11_nn_d.log 2>&1) &
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
until [ -f exp/pred/t1_gpm.npy ] && [ -f exp/pred/p2_leui_wA.npy ]; do sleep 60; done
sleep 30
for v in v13:nn_,nn3_,nn4_,nn5_,n4_,n5_ v13nc:nn_,nn3_,nn4_,nn5_,n4_,n5_,c4_,cat_; do
  tag=${v%%:*}; ex=${v#*:}
  .venv/bin/python -u src/prune.py 25 --exclude $ex > exp/prune_$tag.log 2>&1
  G=$(grep "^gas runs:" exp/prune_$tag.log | cut -d' ' -f3); E=$(grep "^elec runs:" exp/prune_$tag.log | cut -d' ' -f3)
  .venv/bin/python -u src/make_submission.py $tag exp/pred/adopt_nb0.npy --gas $G --elec $E > exp/sub_$tag.log 2>&1
  .venv/bin/python -u src/resid_corr.py apply $tag ${tag}r --lam exp/resid_lam_g.json > exp/sub_${tag}r.log 2>&1
done
echo done >> exp/sub_v13ncr.log
wait
