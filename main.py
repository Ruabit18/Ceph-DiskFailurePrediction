'''
Descripttion: 
Version: xxx
Author: WanJu
Date: 2021-05-21 10:02:33
LastEditors: WanJu
LastEditTime: 2021-05-24 15:14:48
'''
from Ceph_DiskPrediction import Ceph_DP_PS, Ceph_DP_RH
from WNLO_DiskPrediction import WNLO_DP
import os
if __name__ == '__main__':

    mode_attr = ["user_capacity", "smart_1_raw", "smart_5_raw", "smart_7_raw", "smart_10_raw", "smart_187_raw", "smart_188_raw", "smart_190_raw", "smart_193_raw", "smart_197_raw", "smart_198_raw", "smart_241_raw", 
                "smart_1_normalized", "smart_5_normalized", "smart_7_normalized", "smart_10_normalized", "smart_187_normalized", "smart_188_normalized", "smart_190_normalized", "smart_193_normalized", "smart_197_normalized", "smart_198_normalized", "smart_241_normalized"]
    data_path = os.path.join('data', 'ST4000DM000', 'days.csv')
    ceph_dp_rh = Ceph_DP_RH(
        model_attr=mode_attr,
        data_path=data_path
    )
    ceph_dp_rh.run()

    ceph_dp_ps = Ceph_DP_PS(
        data_path=data_path,
        model_path=os.path.join('models', 'prophetstor'),
        config_path=os.path.join('models', 'prophetstor', 'config.json'),
    )
    ceph_dp_ps.run()

    wnlo_dp = WNLO_DP(
        data_path=os.path.join('data', 'ST4000DM000'),
        model_path=os.path.join('models', 'wnlo')
    )
    wnlo_dp.run()