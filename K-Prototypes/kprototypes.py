import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kmodes.kprototypes import KPrototypes
import math
import json
#import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


data_df = pd.read_csv("/home/b07170223/kprototypes/set_3.csv", dtype={'UID':str, 'Gender_Code':str, 'ZIP_Code':str, 'AD_TAG':str, 'INV_time_mode':str, 'APP_time_mode':str, 'TXN_time_mode':str, 'TXN_zip_code':str})


#Standardization
categorical_feature = data_df[['Gender_Code', 'ZIP_Code', 'AD_TAG', 'INV_time_mode', 'APP_time_mode', 'TXN_time_mode', 'TXN_zip_code']]
numerical_feature = data_df.drop(columns=['Gender_Code', 'ZIP_Code', 'AD_TAG', 'INV_time_mode', 'APP_time_mode', 'TXN_time_mode', 'TXN_zip_code'])
data_df[list(numerical_feature.columns)] = StandardScaler().fit_transform(data_df[list(numerical_feature.columns)])
STD_data_df = data_df


#index of categorical columns
categorical_index = list(range(25,32))


#Fitting the data into the model
model_STD = KPrototypes(n_clusters=10, verbose=2, init='Huang', n_jobs=-1)
model_STD.fit_predict(STD_data_df, categorical=categorical_index)
STD_data_df['cluster_labels'] = model_STD.labels_.tolist()
'''
print(STD_data_df['cluster_labels'].value_counts())
print('cost:\n',model_STD.cost_, '\n\n', 
        'weight(a):\n',model_STD.gamma, '\n\n', 
        'cluster_centroids:\n',model_STD.cluster_centroids_
    )
'''

STD_data_df.to_csv("/home/b07170223/kprototypes/result/set3_k10.csv" ,index=False)

output = {"dist_within_cluster": model_STD.cost_,
            "categorical_weight": model_STD.gamma,
            "cluster_centroids": model_STD.cluster_centroids_,
        }

with open("/content/drive/MyDrive/巨量行銷/set3_k10.json", "w") as f:
    json.dump(output, f)
