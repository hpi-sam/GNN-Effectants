#!/bin/bash
echo "Starting training loops for ComplEx..."
for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    for VARIABLE in "body_temp"
    do
        python3 experiment_ComplEx.py -se ${VARIABLE} > "results/ComplEx_${VARIABLE}_${INDEX}.txt"
    done
done
