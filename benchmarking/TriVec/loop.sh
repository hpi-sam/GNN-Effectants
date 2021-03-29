#!/bin/bash
echo "Starting training loops for TriVec..."
for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    for VARIABLE in "mumps" "emesis" "bleeding" "body_temp" "coccydynia" "carbuncle"
    do
        python3 experiment_TriVec.py -se ${VARIABLE} > "results/TriVec_${VARIABLE}_${INDEX}.txt"
    done
done
