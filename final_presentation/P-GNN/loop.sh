#!/bin/bash
echo "Starting training loops for P-GNN..."
for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    for VARIABLE in "mumps" "emesis" "bleeding" "body_temp" "coccydynia" "carbuncle"
    do
        python3 main.py --model PGNN --layer_num 3 --dataset ${VARIABLE} --cpu > "PGNN_${VARIABLE}_${INDEX}.txt"
    done
done
