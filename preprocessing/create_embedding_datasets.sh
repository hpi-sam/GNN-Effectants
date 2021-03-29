echo "Starting creation of training and test and validation data..."
for SIDEEFFECT in "Mumps" "emesis" "Bleeding" "body_temp" "coccydynia" "carbuncle"
do
    python3 preprocessing_embedding.py -ef ../dataframe_top3_final.csv -se ${SIDEEFFECT} -td data -tts 0.3

    gzip "data/${SIDEEFFECT}_train.txt"
    gzip "data/${SIDEEFFECT}_test.txt"
    gzip "data/${SIDEEFFECT}_val.txt"
done