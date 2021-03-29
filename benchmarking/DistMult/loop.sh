#!/bin/bash
echo "Starting training loops for DistMult..."
for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    for VARIABLE in "mumps" "emesis" "bleeding" "body_temp" "coccydynia" "carbuncle"
    do
        python3 experiment_DistMult.py -se ${VARIABLE} > "results/DistMult_${VARIABLE}_${INDEX}.txt"
    done
done
