# -*- encoding: utf-8 -*-
"""
@File        :  scoring.py
@Author      :  WanJu
@Contact     :  qq: 627866757 / wx: ruabit18
@Modify Time :  2021/4/15 20:46
@Version     :  3.0
@Description :
"""
import numpy as np
import prettytable as pt
from sklearn import metrics
from enum import Enum, unique


@unique
class Scoring(Enum):
    Accuracy = 0        # 正确率（Accuracy） 被正确分类的样本比例或数量
    Recall = 1          # 召回率（recall） 分类器预测为正例的样本占实际正例样本数量的比例，描述了分类器对正例类别的敏感程度
    Specificity = 2     # 特异性（Specificity） 负类正确预测率
    Error_Rate = 3      # 错误率（Error Rate） 被错误分类的样本比例或数量
    FAR = 4             # 误报率（failure alarm rate）
    FNR = 5             # 漏报率
    Precision = 6       # 精度（Precision） 在所有判别为正例的结果中，真正正例所占的比例。
    AUC = 7             # 计算AUC


class Score:
    def __init__(self):
        pass

    @staticmethod
    def check_matrix(confusion_matrix):
        if np.size(confusion_matrix) != 4:
            return False
        if confusion_matrix[0][0] == 0 and confusion_matrix[0][1] == 0:
            return False
        if confusion_matrix[1][0] == 0 and confusion_matrix[1][1] == 0:
            return False
        return True


    @staticmethod
    def calculate_matrix(confusion_matrix, y_true, y_score):
        """
        计算混淆矩阵
        @param confusion_matrix:模型测试结果的混淆矩阵
        @param y_true:真实的标签数据
        @param y_score:模型预测的标签数据
        @return:
        """
        if not Score.check_matrix(confusion_matrix):
            return np.zeros(8)

        TP = confusion_matrix[1][1]
        FN = confusion_matrix[1][0]
        FP = confusion_matrix[0][1]
        TN = confusion_matrix[0][0]
        # 计算模型指标
        Accuracy = 0 if (TP + FN + FP + TN) == 0 else (TP + TN) / (TP + FN + FP + TN)  # 正确率（Accuracy） 被正确分类的样本比例或数量
        Recall = 0 if (TP + FN) == 0 else TP / (TP + FN)  # 召回率（recall） 分类器预测为正例的样本占实际正例样本数量的比例，描述了分类器对正例类别的敏感程度
        Specificity = 0 if (TN + FP) == 0 else TN / (TN + FP)  # 特异性（Specificity） 负类正确预测率
        Error_Rate = 0 if (TP + FN + FP + TN) == 0 else (FP + FN) / (TP + FN + FP + TN)  # 错误率（Error Rate） 被错误分类的样本比例或数量
        FAR = 0 if (FP + TN) == 0 else FP / (FP + TN)  # 误报率（failure alarm rate）
        FNR = 0 if (TP + FN) == 0 else FN / (TP + FN)  # 漏报率
        Precision = 0 if (TP + FP) == 0 else TP / (TP + FP)  # 精度（Precision） 在所有判别为正例的结果中，真正正例所占的比例。
        AUC = metrics.roc_auc_score(y_true, y_score)  # 计算AUC

        return [Accuracy, Recall, Specificity, Error_Rate, FAR, FNR, Precision, AUC]


    @staticmethod
    def model_score(confusion_matrix, y_true, y_score, scoring):
        """
        输出模型分数
        @param confusion_matrix:模型测试结果的混淆矩阵
        @param y_true:真实的标签数据
        @param y_score:模型预测的标签数据
        @param scoring:所选择的模型指标
        @return:
        """
        return Score.calculate_matrix(confusion_matrix, y_true, y_score)[scoring]

    @staticmethod
    def print_confusion_matrix(confusion_matrix, y_true, y_score):
        if not Score.check_matrix(confusion_matrix):
            return
        TP = confusion_matrix[1][1]
        FN = confusion_matrix[1][0]
        FP = confusion_matrix[0][1]
        TN = confusion_matrix[0][0]
        tb = pt.PrettyTable()
        tb.padding_width = 1
        tb.hrules = pt.FRAME
        tb.vrules = pt.FRAME
        tb.field_names = ['', ' ', 'Predict', '  ']
        tb.align = 'l'
        tb.add_row([' ', ' ', '1', '0'])
        tb.add_row(['Real', '1：' + str(TP + FN), 'TP:' + str(TP), 'FN:' + str(FN)])
        tb.add_row(['', '0：' + str(FP + TN), 'FP:' + str(FP), 'TN:' + str(TN)])
        print(tb)

        scores = Score.calculate_matrix(confusion_matrix, y_true, y_score)

        result = pt.PrettyTable()
        result.field_names = ['Target', 'Value']
        result.padding_width = 1
        result.align = 'l'
        result.add_row(['FDR', scores[Scoring.Recall.value]])
        result.add_row(['FAR', scores[Scoring.FAR.value]])
        result.add_row(['AUC', scores[Scoring.AUC.value]])
        result.add_row(['FNR', scores[Scoring.FNR.value]])
        result.add_row(['Accuracy', scores[Scoring.Accuracy.value]])
        result.add_row(['Precision', scores[Scoring.Precision.value]])
        result.add_row(['Specificity', scores[Scoring.Specificity.value]])
        result.add_row(['Error Rate', scores[Scoring.Error_Rate.value]])
        print(result)
