# -*- coding:utf-8 -*-
"""
@contact: zhangyf.dalian@gmail.com
@time: 2020-01-09
"""
__author__ = 'zhangyf'

from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from zhangyifei.config import *
from zhangyifei.data_utils import DataSet

TASK_LIST = ['age', 'gender', 'education']
fasttext_dataset = DataSet(Path(data_dir, 'fasttext'))
cv_dataset = DataSet(Path(data_dir, 'cross_validation'))
stacking_dataset = DataSet(Path(data_dir, 'stacking'))


def fasttext_model():
    """对三组任务(age, gender education)分别执行5折交叉验证，
    保存预测的概率值到`config.data_dir`/stacking路径
    保存验证集准确率到`config.data_dir`/fasttext/fasttest_cv_accuracy.csv文件"""
    data_path = Path(data_dir, 'fasttext')
    model_path = Path(model_dir)
    if not model_path.exists():
        model_path.mkdir()

    df_res = pd.DataFrame(index=range(5), columns=TASK_LIST)

    for task in TASK_LIST:
        for i in range(5):
            # 训练fasttext
            train_file = str(data_path / f'{task}_train_cv{i}.txt')
            clf = fasttext.train_supervised(train_file)
            model_file = str(Path(model_path, f'fasttext_model_{task}_{i}.bin'))
            clf.save_model(model_file)

            # 预测训练集
            input_file = f'{task}_train_cv{i}.txt'
            output_prob_file = f'prob_{task}_train_cv{i}.csv'
            _, _ = fasttext_pred(clf, input_file, output_prob_file)

            # 预测验证集
            input_file = f'{task}_valid_cv{i}.txt'
            output_prob_file = f'prob_{task}_valid_cv{i}.csv'
            valid_label_pred, valid_label_true = fasttext_pred(
                clf, input_file, output_prob_file)

            # 评估
            acc = metrics.accuracy_score(valid_label_true, valid_label_pred)
            df_res.loc[i, task] = acc
    fasttext_dataset.df_to_csv('fasttest_cv_accuracy.csv', df_res)


def fasttext_pred(clf, fasttext_file, output_prob_file):
    """
    执行fasttext预测，返回预测的标签值与真实的标签值，
    预测概率保存到文件`output_prob_file`

    :param clf: fasttext.train_supervised训练的分类器
    :param fasttext_file: 符合fasttext格式的输入文件路径
    :param output_prob_file: 预测概率值的保存路径
    :return:
        label_pred: 预测的标签值
        label_true: 真实的标签值

    """
    valid_text, label_true = fasttext_dataset.read_fasttext_file(fasttext_file)
    label_pred, prob = fasttext_pred_prob(clf, valid_text)
    df_prob = pd.DataFrame(prob,
                           columns=[f'prob_{j}' for j in range(prob.shape[1])])
    stacking_dataset.df_to_csv(output_prob_file, df_prob)
    return label_pred, label_true


def fasttext_pred_prob(clf, text):
    """执行fasttext概率预测，返回预测的标签及概率矩阵"""
    label_pred, prob = clf.predict(text, k=10)
    prob = sort_prob(label_pred, prob)
    label_pred = [int(i[0].replace('__label__', '')) for i in
                  label_pred]
    return label_pred, prob


def sort_prob(labels, prob):
    """fasttext预测的概率是按照其数值大小降序排列的，将其改为按标签顺序排列"""
    labels = [[int(j.replace('__label__', '')) for j in i] for i in labels]
    idx = np.argsort(labels, axis=1)
    sorted_prob = np.take_along_axis(prob, idx, axis=1)
    return sorted_prob


def generate_xgb_features():
    """合并fasttext预测的概率，生成xgb模型的输入文件"""
    for i in range(5):
        train_data_list, valid_data_list = [], []
        for task in TASK_LIST:
            train_file = f'prob_{task}_train_cv{i}.csv'
            valid_file = f'prob_{task}_valid_cv{i}.csv'
            df_train_ = stacking_dataset.read_csv(train_file)
            df_valid_ = stacking_dataset.read_csv(valid_file)
            columns = [f'{task}_{i}' for i in df_train_.columns]
            df_train_.columns = columns
            df_valid_.columns = columns
            train_data_list.append(df_train_)
            valid_data_list.append(df_valid_)
        for j in range(len(train_data_list) - 1):
            assert len(train_data_list[j]) == len(train_data_list[j + 1])
        for j in range(len(valid_data_list) - 1):
            assert len(valid_data_list[j]) == len(valid_data_list[j + 1])
        df_train = pd.concat(train_data_list, axis=1)
        df_valid = pd.concat(valid_data_list, axis=1)

        stacking_dataset.df_to_csv(f'xgb_train_cv{i}.csv', df_train)
        stacking_dataset.df_to_csv(f'xgb_valid_cv{i}.csv', df_valid)


def xgb_model():
    xgb_params = {
        'learning_rate': 0.01,
        'n_estimators': 100,
        'max_depth': 5,
        'min_child_weight': 0,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 2020,
        'nthread': 2,
        'gamma': 0.0,
        # 'colsample_bylevel': 1.0,
        # 'reg_alpha': 0.9,
        # 'reg_lambda': 1.0,
    }
    df_eval = pd.DataFrame(index=range(5), columns=TASK_LIST)
    for task in TASK_LIST:
        print(f'xgb model, {task} task')
        for i in range(5):
            label_train = cv_dataset.read_list(f'{task}_train_cv{i}.csv')
            label_valid = cv_dataset.read_list(f'{task}_valid_cv{i}.csv')
            fea_train = stacking_dataset.read_csv(f'xgb_train_cv{i}.csv')
            fea_valid = stacking_dataset.read_csv(f'xgb_valid_cv{i}.csv')
            xgb = XGBClassifier(**xgb_params)
            xgb.fit(X=fea_train, y=label_train)
            label_valid_pred = xgb.predict(fea_valid)
            acc = metrics.accuracy_score(label_valid, label_valid_pred)
            df_eval.loc[i, task] = acc
            print(f'cv {i}, accuracy: {acc}')
    stacking_dataset.df_to_csv(f'xgb_eval.csv', df_eval)


def lgb_model():
    lgb_params = {
        "boosting_type": "gbdt",
        "num_leaves": 255,
        "learning_rate": 0.01,
        "n_estimators": 200,
        "objective": "multiclass",
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "random_state": 2020,
    }
    df_eval = pd.DataFrame(index=range(5), columns=TASK_LIST)
    for task in TASK_LIST:
        print(f'lgb model, {task} task')
        for i in range(5):
            label_train = cv_dataset.read_list(f'{task}_train_cv{i}.csv')
            label_valid = cv_dataset.read_list(f'{task}_valid_cv{i}.csv')
            fea_train = stacking_dataset.read_csv(f'xgb_train_cv{i}.csv')
            fea_valid = stacking_dataset.read_csv(f'xgb_valid_cv{i}.csv')
            lgb = LGBMClassifier(**lgb_params)
            lgb.fit(X=fea_train, y=label_train)
            label_valid_pred = lgb.predict(fea_valid)
            acc = metrics.accuracy_score(label_valid, label_valid_pred)
            df_eval.loc[i, task] = acc
            print(f'cv {i}, accuracy: {acc}')
    stacking_dataset.df_to_csv(f'lgb_eval.csv', df_eval)


def main():
    fasttext_model()
    generate_xgb_features()
    xgb_model()
    # lgb_model()


if __name__ == '__main__':
    main()
