#!/bin/bash
# after the v8 XGB members finish: re-select blend members with them as candidates and build sub_v8
cd /root/Dicefest
while kill -0 2534858 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/prune.py 25 --exclude nn_,nn3_,nn4_,nn5_,n4_,n5_ > exp/prune_v8.log 2>&1
G=$(grep "^gas runs:" exp/prune_v8.log | cut -d' ' -f3); E=$(grep "^elec runs:" exp/prune_v8.log | cut -d' ' -f3)
.venv/bin/python -u src/make_submission.py v8 exp/pred/adopt_nb0.npy --gas $G --elec $E > exp/sub_v8.log 2>&1
echo done >> exp/sub_v8.log
