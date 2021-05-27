'''
Descripttion: 
Version: xxx
Author: WanJu
Date: 2021-05-21 17:14:16
LastEditors: WanJu
LastEditTime: 2021-05-21 18:12:21
'''
import numpy as np
mode_attr = ["user_capacity", "smart_11_raw", "smart_51_raw"]
mode_attr2 = ["user_capacity", "smart_1_raw", "smart_5_raw"]
l1 = [attr in mode_attr for attr in mode_attr2]
print(l1)
print(sum([attr in mode_attr for attr in mode_attr2]))
