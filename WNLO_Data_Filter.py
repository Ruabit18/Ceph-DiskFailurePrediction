'''
Descripttion: 
Version: xxx
Author: WanJu
Date: 2021-05-21 11:48:37
LastEditors: WanJu
LastEditTime: 2021-05-21 16:18:15
'''
import pandas as pd
import os


data_path = 'original_data/'
model = 'ST4000DM000'

device_dict = {}
index_failure = 4
index_data = 5

all_data = pd.DataFrame()
for file in os.listdir(data_path):
    print('\rprocessing:' + file, end='')
    df = pd.read_csv(os.path.join(data_path, file))
    df = df[df['model'] == model].iloc[:, index_failure:]
    error_data = df[df['failure'] == 1]
    normal_data = df[df['failure'] == 0].sample(random_state=18, n=error_data.shape[0] * 3)
    all_data = pd.concat([all_data, error_data, normal_data], axis=0, ignore_index=True)

if (not os.path.exists(os.path.join('data', model))):
    os.makedirs(os.path.join('data', model))

all_data.dropna(axis=1, how='all', inplace=True)
all_data.to_csv(os.path.join('data', model, 'test.csv'), header=True, index=False)
