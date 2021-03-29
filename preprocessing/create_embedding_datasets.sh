echo "Starting creation of training and test and validation data..."
for SIDEEFFECT in #names of python input
do
    #python command
    gzip "data/${SIDEEFFECT}_train.txt"
    gzip "data/${SIDEEFFECT}_test.txt"
    gzip "data/${SIDEEFFECT}_val.txt"
echo "Removing temporary txts..."
    rm "data/${SIDEEFFECT}_train.txt"
    rm "data/${SIDEEFFECT}_test.txt"
    rm "data/${SIDEEFFECT}_val.txt" 
done