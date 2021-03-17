#!/bin/bash
echo "Starting training loops for SEAL with embedding..."
for VARIABLE in "emesis"
do
     for INDEX in 1 2 3 4 5 6 7 8 9 10
    do
	    python Main.py --train-name "${VARIABLE}_train.txt" --test-name "${VARIABLE}_test.txt" --no-parallel --use-embedding > "SEAL_embedding_${VARIABLE}_${INDEX}.txt"
    done
done

echo "Starting training loops for SEAL without embedding..."
for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    for VARIABLE in "Mumps" "carbuncle" "coccydynia" "Bleeding" "body temperature increased" "emesis"
    do
	    python Main.py --train-name "${VARIABLE}_train.txt" --test-name "${VARIABLE}_test.txt" --no-parallel > "SEAL_${VARIABLE}_${INDEX}.txt"
    done
done
