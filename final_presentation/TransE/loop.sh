#!/bin/bash
echo "Starting training loops for TransE..."
for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    for VARIABLE in "mumps" "emesis" "bleeding" "body_temp" "coccydynia" "carbuncle"
    do
        python3 experiment_TransE.py -se ${VARIABLE} > "TransE_${VARIABLE}_${INDEX}.txt"
    done
done
