import pandas as pd
import numpy as np

def preprocess_se_data(save_path, data_path="dataframe_top3_new.csv"):
    data=pd.read_csv(data_path)
    
    for name in ["Mumps","emesis","Bleeding","body_temp","coccydynia","carbuncle"]:
        data_values=data.loc[data.label==name]
        data_values=data_values[["source","target"]]
        data_values.to_csv(f'{save_path}/data_{name}.txt', header=False, index=False, sep=' ', mode='a')
        
        data_label=data.loc[data.label==name].label
        data_label=pd.DataFrame(np.ones(max(data_label.index),dtype=int))
        data_label.to_csv(f'{save_path}/labels_{name}.txt', header=False, index=True, sep=' ', mode='a')
    
    