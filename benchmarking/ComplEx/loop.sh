#!/bin/bash
echo "Starting training loops for ComplEx..."
for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    for VARIABLE in "mumps" "emesis" "bleeding" "body_temp" "coccydynia" "carbuncle"
    do
        python3 experiment_ComplEx.py -se ${VARIABLE} > "ComplEx_${VARIABLE}_${INDEX}.txt"
    done
done
