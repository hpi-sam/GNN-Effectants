import argparse
import pandas as pd
import numpy as np


######################################
#        COMMAND LINE PARAMS        #
#####################################

parser = argparse.ArgumentParser(description='Set necessary values to train different types of predefined models')
parser.add_argument("--edgefile", "-ef", help="set path to csv containing labelled edges")
parser.add_argument("--sideeffect", "-se", help="specify name of side-effect to extract")
parser.add_argument("--targetdir", "-td", help="set path to the target directory")
parser.add_argument("--valtestsplit", "-tts", help="set the train-test split")

args = parser.parse_args()

data_full=pd.read_csv(args.edgefile)

data_part=data_full.loc[data_full["label"]==args.sideeffect]

from sklearn.model_selection import train_test_split
drug_train, drug_test = train_test_split(data_part, test_size=float(args.valtestsplit), random_state=42)
drug_test, drug_val = train_test_split(drug_test, test_size=0.5, random_state=42)

print('Number of training edges: {}'.format(len(drug_train)))
print('Number of test edges: {}'.format(len(drug_test)))
print('Number of val edges: {}'.format(len(drug_val)))

np.savetxt('{path}/{filename}_train.txt'.format(path=args.targetdir, filename=args.sideeffect), drug_train.values, fmt='%s', delimiter=" ") 
np.savetxt('{path}/{filename}_test.txt'.format(path=args.targetdir, filename=args.sideeffect), drug_test.values, fmt='%s', delimiter=" ") 
np.savetxt('{path}/{filename}_val.txt'.format(path=args.targetdir, filename=args.sideeffect), drug_val.values, fmt='%s', delimiter=" ") 