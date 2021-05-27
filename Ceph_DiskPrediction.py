'''
Descripttion: 
Version: xxx
Author: WanJu
Date: 2021-05-21 10:02:52
LastEditors: WanJu
LastEditTime: 2021-05-24 11:56:43
'''
import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.metrics import confusion_matrix
from score import Score, Scoring
import pandas as pd
import datetime
import pickle
import json
import os


class Ceph_DP_RH:
    def __init__(self, data_path, model_attr) -> None:
        self.data_path_ = data_path
        self.model_attr_ = model_attr
        self.data_ = pd.DataFrame()
        self.device_id_ = []

    def run(self):
        self.getData()
        self.predict()

    @staticmethod
    def time_transform(time, old_form, new_form):
        date = datetime.datetime.strftime(time, old_form)
        return datetime.datetime.strptime(date, new_form)
        
    def getData(self):
        print('\n### 获取数据...')
        self.data_ = pd.DataFrame(pd.read_csv(self.data_path_, encoding='utf-8'))
        self.data_.sort_values(by='date', inplace=True)
        # print(error_id)
        self.device_id_ = list(self.data_['serial_number'].value_counts().to_dict().keys())
        print('共%d块硬盘.' % len(self.device_id_))

    def preProcess(self, dick_days):
        struct_type = [(attr, np.float64) for attr in self.model_attr_]
        values = [tuple(day[attr] for attr in self.model_attr_) for day in dick_days]
        disk_days_sa = np.array(values, dtype=struct_type)
        disk_days_attrs = disk_days_sa[[attr for attr in self.model_attr_ if 'smart_' in attr]]
        disk_days_attrs = np.array([list(i) for i in disk_days_attrs])
        
        roll_window_size = 6

        gen = [disk_days_attrs[i: i + roll_window_size, ...].mean(axis=0) \
                for i in range(0, disk_days_attrs.shape[0] - roll_window_size + 1)]
        means = np.vstack(gen)
        gen = [disk_days_attrs[i: i + roll_window_size, ...].std(axis=0, ddof=1) \
                for i in range(0, disk_days_attrs.shape[0] - roll_window_size + 1)]
        stds = np.vstack(gen)
        cvs = stds / means
        cvs[np.isnan(cvs)] = 0
        featurized = np.hstack((
                                means,
                                stds,
                                cvs,
                                disk_days_sa['user_capacity'][: disk_days_attrs.shape[0] - roll_window_size + 1].reshape(-1, 1)
                                ))
        with open('models/redhat/seagate_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        featurized = scaler.transform(featurized)
        return featurized
    
    def predict(self):
        pred_class = []
        true_class = []
        num = 0
        with open('models/redhat/seagate_predictor.pkl', 'rb') as f:
            model = pickle.load(f)

        for device_id in list(self.device_id_):
            num += 1
            dick_days = []
            failure = 0
            dick_df = self.data_[self.data_['serial_number'] == device_id]
            print('\r device_id:', device_id, num, '/', len(self.device_id_), end='')
            for row in dick_df.itertuples():
                pre_data = {}
                pre_data['failure'] = getattr(row, 'failure')
                if pre_data['failure'] == 1:
                    failure = 1
                pre_data['date'] = getattr(row, 'date')
                pre_data['serial_number'] = getattr(row, 'serial_number')
                pre_data['model'] = getattr(row, 'model')
                for attr in self.model_attr_:
                    pre_data[attr] = getattr(row, attr)
                dick_days.append(pre_data)
            
            processed_data = self.preProcess(dick_days=dick_days)
            pred_class_id = model.predict(processed_data)[-1]
            pred_class.append(pred_class_id)
            true_class.append(failure)
        
        # print('pred_class:')
        # print(pred_class)
        # print('-1', pred_class.count(-1))
        # print('0', pred_class.count(0))
        # print('1', pred_class.count(1))
        # print('2', pred_class.count(2))
        
        print('true_class:')
        print(true_class)
        # print('0', true_class.count(0))
        # print('1', true_class.count(1))

        print('pred_class:')
        pred_class = [1 if i != 0 else 0 for i in pred_class]
        print(pred_class)
        print("\nCeph_DP_RH 模型评估:")
        martix_ = confusion_matrix(y_pred=pred_class, y_true=true_class)
        Score.print_confusion_matrix(y_score=pred_class, y_true=true_class, confusion_matrix=martix_)


class Ceph_DP_PS:
    def __init__(self, data_path, model_path, config_path) -> None:
        with open(config_path) as file:
            self.model_content_ = json.load(file)
        self.model_path_ = model_path
        self.data_path_ = data_path
        self.model_attr_ = []
        self.data_ = pd.DataFrame()
        self.device_id_ = []

    def run(self):
        self.getData()
        self.predict()

    def getData(self):
        print('\n### 获取数据...')
        self.data_ = pd.DataFrame(pd.read_csv(self.data_path_, encoding='utf-8'))
        self.data_.sort_values(by='date', inplace=True)
        self.model_attr_ = [attr for attr in self.data_.columns if (attr.startswith("smart_") and attr.endswith("_raw"))]
        self.device_id_ = list(self.data_['serial_number'].value_counts().to_dict().keys())
        print('共%d块硬盘.' % len(self.device_id_))

    def preProcess(self, disk_days):
        diff_disk_days = self.getDiff(disk_days)
        model_list = self.getBestMode()
        return model_list, diff_disk_days
        
    def getDiff(self, disk_days):
        prev_days = disk_days[:-1]
        curr_days = disk_days[1:]
        diff_disk_days = []
        # TODO: ensure that this ordering is correct
        for prev, cur in zip(prev_days, curr_days):
            diff_disk_days.append(
                {attr: (int(cur[attr]) - int(prev[attr])) for attr in self.model_attr_}
            )
        return diff_disk_days

    def getBestMode(self):
        models = self.model_content_.keys()
        scores = []
        for model_name in models:
            scores.append(
                sum(attr in self.model_attr_ for attr in self.model_content_[model_name])
            )
        max_score = max(scores)
        # Skip if too few matched attributes.
        if max_score < 3:
            print("Too few matched attributes")
            return None
        best_models = {}
        best_model_indices = [
            idx for idx, score in enumerate(scores) if score > max_score - 2
        ]
        for model_idx in best_model_indices:
            model_name = list(models)[model_idx]
            model_path = os.path.join(self.model_path_, model_name)
            model_attrlist = self.model_content_[model_name]
            best_models[model_path] = model_attrlist

        return best_models

    @staticmethod
    def getOrderData(disk_days, model_attr):
        ordered_attrs = []
        for one_day in disk_days:
            one_day_attrs = []
            for attr in model_attr:
                if attr in one_day:
                    one_day_attrs.append(one_day[attr])
                else:
                    one_day_attrs.append(0)
            ordered_attrs.append(one_day_attrs)
        return ordered_attrs

    def predict(self):
        pred_class = []
        true_class = []
        models = []
        num = 0

        for device_id in list(self.device_id_):
            num += 1
            dick_days = []
            failure = 0
            dick_df = self.data_[self.data_['serial_number'] == device_id]
            print('\r device_id:', device_id, num, '/', len(self.device_id_), end='')
            for row in dick_df.itertuples():
                pre_data = {}
                pre_data['failure'] = getattr(row, 'failure')
                if pre_data['failure'] == 1:
                    failure = 1
                pre_data['date'] = getattr(row, 'date')
                pre_data['serial_number'] = getattr(row, 'serial_number')
                pre_data['model'] = getattr(row, 'model')
                for attr in self.model_attr_:
                    pre_data[attr] = getattr(row, attr)
                dick_days.append(pre_data)

            model_list, diff_disk_days = self.preProcess(dick_days)
            if (len(models) <= 0):
                for modelpath in model_list:
                    model_attrlist = model_list[modelpath]
                    with open(modelpath, "rb") as f_model:
                        models.append([pickle.load(f_model, encoding="latin1"), model_attrlist])
            
            all_pred = []
            for model in models:
                ordered_data = Ceph_DP_PS.getOrderData(diff_disk_days, model[1])
                pred = model[0].predict(ordered_data)
                all_pred.append(1 if any(pred) else 0)
            
            pred_class_id = 1 if 2 ** sum(all_pred) - len(models) > 4 else 0
            pred_class.append(pred_class_id)
            true_class.append(failure)
        
        # print('\npred_class:')
        # print(pred_class)
        # print('-1', pred_class.count(-1))
        # print('0', pred_class.count(0))
        # print('1', pred_class.count(1))
        # print('2', pred_class.count(2))
        
        print('true_class:')
        print(true_class)
        # print('0', true_class.count(0))
        # print('1', true_class.count(1))

        print('\npred_class:')
        pred_class = [1 if i != 0 else 0 for i in pred_class]
        print(pred_class)
        print('best_model_list:', len(models))
        print("\nCeph_DP_PS 模型评估:")
        martix_ = confusion_matrix(y_pred=pred_class, y_true=true_class)
        Score.print_confusion_matrix(y_score=pred_class, y_true=true_class, confusion_matrix=martix_)