#!/bin/bash
# kNN residual correction: holdout fit of the weights, then apply to v15nc / v15 (+ dup_fix)
cd /root/Dicefest
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
.venv/bin/python -u src/resid_knn.py fit --lam exp/resid_knn.json > exp/resid_knn_fit.log 2>&1 || exit 1
for tag in v15nc v15; do
  .venv/bin/python -u src/resid_knn.py apply $tag ${tag}rk --lam exp/resid_knn.json > exp/sub_${tag}rk.log 2>&1 || exit 1
  .venv/bin/python -u src/dup_fix.py ${tag}rk ${tag}rkd >> exp/sub_${tag}rk.log 2>&1
done
echo done >> exp/sub_v15rk.log
