'''
Descripttion: 
Version: xxx
Author: WanJu
Date: 2021-05-21 11:48:37
LastEditors: WanJu
LastEditTime: 2021-05-24 10:24:47
'''
import pandas as pd
import os


data_path = 'original_data/'
model = 'ST4000DM000'

device_dict = {}
days = 6

all_data = pd.DataFrame()
for file in os.listdir(data_path):
    print('\rprocessing:' + file, end='')
    df = pd.read_csv(os.path.join(data_path, file))
    df = df[df['model'] == model]
    for row in df.itertuples():
        sn = getattr(row, 'serial_number')
        failure = getattr(row, 'failure')
        if (failure == 1):
            if sn in device_dict.keys() and device_dict[sn][0] >= days:
                device_dict[sn][1] = 1
        else:
            if sn not in device_dict.keys():
                device_dict[sn] = [0, 0]
            device_dict[sn][0] += 1
    all_data = pd.concat([all_data, df], axis=0, ignore_index=True)

error_id = [sn for sn in device_dict.keys() if device_dict[sn][1] == 1]
normal_id = [sn for sn in device_dict.keys() if (device_dict[sn][1] == 0 and device_dict[sn][0] > days)][:len(error_id) * 3]
print('\n', len(error_id))
useful_id = error_id + normal_id

if (not os.path.exists(os.path.join('data', model))):
    os.makedirs(os.path.join('data', model))
all_data = all_data[all_data['serial_number'].isin(useful_id)]
all_data.dropna(axis=1, how='all', inplace=True)
all_data.rename(columns={'capacity_bytes':'user_capacity'}, inplace=True)
all_data.to_csv(os.path.join('data', model, 'days.csv'), header=True, index=False)
