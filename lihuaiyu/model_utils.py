# -*- coding:utf-8 -*-
import json


def generate_train_data(data_df, group_name="Gender", k_fold_attr_name='gender_kfold_index', n=0):
    ''' n=0时 第一折 n=2时第二折 ......'''
    data_df = data_df[data_df[group_name] != 0]
    train = data_df[data_df[k_fold_attr_name] != n]
    train_data = train['Query'].values
    train_label = train[group_name].values
    val = data_df[data_df[k_fold_attr_name] == n]
    val_data = val['Query'].values
    val_label = val[group_name].values
    return train_data, train_label, val_data, val_label




def read_json(data_path):
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict


def to_json(json_path, data_dict):
    with open(json_path, 'w') as f:
        json.dump(data_dict, f)
    print(f'{json_path} 已保存！')
