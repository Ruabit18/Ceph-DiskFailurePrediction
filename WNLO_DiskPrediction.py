'''
Descripttion: 
Version: xxx
Author: WanJu
Date: 2021-05-21 10:03:21
LastEditors: WanJu
LastEditTime: 2021-05-24 15:13:34
'''
from numpy.lib.function_base import select
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from score import Score
import pickle
import pandas as pd
import os, sys

pd.set_option('display.max_rows', 10, 'display.max_columns', 1000)

class WNLO_DP:
    def __init__(self, data_path, model_path) -> None:
        self.data_path_ = data_path
        self.model_path_ = model_path

        self.feature_ = []
        self.model_ = RandomForestClassifier()
        self.train_data_ = pd.DataFrame()
        self.test_data_ = pd.DataFrame()
        

    def run(self):
        self.getTestData()
        self.getTrainData()
        self.predict()

    def getTrainData(self):
        self.train_data_ = pd.read_csv(os.path.join(self.data_path_, 'train.csv'))

    def getTestData(self):
        self.test_data_ = pd.read_csv(os.path.join(self.data_path_, 'test.csv'))

    def predict(self):
        if (not os.path.exists(os.path.join(self.model_path_ , 'feature.pkl')) or \
            not os.path.exists(os.path.join(self.model_path_ , 'rf.pkl')) or \
            not os.path.exists(os.path.join(self.model_path_ , 'scale.pkl'))):
            self.train()

        print("WNLO 模型评估:")
        with open(os.path.join(self.model_path_ , 'rf.pkl'), 'rb')as file:
            self.model_ = pickle.load(file)
        with open(os.path.join(self.model_path_ , 'scale.pkl'), 'rb')as file:
            scale = pickle.load(file)
        with open(os.path.join(self.model_path_ , 'feature.pkl'), 'rb')as file:
            self.feature_ = list(pickle.load(file))

        test_x = self.test_data_.iloc[:, 1:]
        test_y = self.test_data_.iloc[:, 0]
        test_x = pd.DataFrame(scale.transform(test_x)).iloc[:, self.feature_]
        pred_y = self.model_.predict(test_x)
        matrix_ = confusion_matrix(y_pred=pred_y, y_true=test_y)
        Score.print_confusion_matrix(y_score=pred_y, y_true=test_y, confusion_matrix=matrix_)
    
    def train(self):
        print("模型不存在, 开始模型训练:")
        self.model_ = RandomForestClassifier()

        train_x, test_x, train_y, test_y = \
                train_test_split(self.train_data_.iloc[:, 1:], self.train_data_.iloc[:, 0], test_size=0.3, random_state=18)

        ss = StandardScaler()
        train_x = pd.DataFrame(ss.fit_transform(train_x))
        
        clf = RandomForestClassifier(n_estimators=50)
        clf = clf.fit(train_x, train_y)
        sfm = SelectFromModel(
            estimator=clf,
            prefit=True
        )
        print('特征重要性:', )
        print(clf.feature_importances_)
        self.feature_ =sfm.get_support(indices=True)
        # self.feature_ = [3, 6, 7, 8, 11, 15, 18, 21, 22, 28, 30, 37, 40, 45, 46, 47]
        print("\nbest_feature:", self.feature_, end='\n\n')
        print(list(range(3, len(self.feature_) + 1, int((len(self.feature_)-3) / 4))))
        param = {
            'max_features':list(range(3, len(self.feature_) + 1, int((len(self.feature_)-3) / 4))),
            'max_depth':[18, 19, 16],
            'n_estimators':[27, 32, 35]
        }
        gscv = GridSearchCV(
            cv=5,
            param_grid=param,
            estimator=self.model_,
            scoring='roc_auc',
            n_jobs=1,
            verbose=2
        )
        gscv.fit(X=train_x.iloc[:, self.feature_], y=train_y)
        self.model_ = gscv.best_estimator_
        print("best_param:", gscv.best_params_)
        print("best_score:", gscv.best_score_)

        print("模型评估:")
        test_x = pd.DataFrame(ss.transform(test_x))
        pred_y = self.model_.predict(test_x.iloc[:, self.feature_])
        matrix_ = confusion_matrix(y_pred=pred_y, y_true=test_y)
        Score.print_confusion_matrix(y_score=pred_y, y_true=test_y, confusion_matrix=matrix_)
        print("模型保存:", end='')

        if (not os.path.exists(self.model_path_)):
            os.makedirs(self.model_path_)
        
        with open(os.path.join(self.model_path_ , 'rf.pkl'), 'wb')as file:
            pickle.dump(self.model_, file)
        with open(os.path.join(self.model_path_ , 'scale.pkl'), 'wb')as file:
            pickle.dump(ss, file)
        with open(os.path.join(self.model_path_ , 'feature.pkl'), 'wb')as file:
            pickle.dump(self.feature_, file)

        print("完成！")

        
