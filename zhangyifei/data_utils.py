# -*- coding:utf-8 -*-
"""
@contact: zhangyf.dalian@gmail.com
@time: 2020-01-09
"""
__author__ = 'zhangyf'

import os
from pathlib import Path

import pandas as pd


class DataSet(object):
    """数据读取有关的类"""

    def __init__(self, data_dir):
        """

        :param data_dir: 数据文件根目录
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir()

    def read_raw_data(self, filename, set_type='train'):
        """
        读取原始数据train.csv和test.csv

        :param filename: 文件名
        :param set_type: 标记是否为训练集或测试集
        :return: pd.DataFrame

        """
        columns_map = {
            'train': ['id', 'age', 'gender', 'education', 'query'],
            'test': ['id', 'query'],
        }
        filepath = self.data_dir / filename
        df = pd.read_csv(filepath, delimiter='###__###', header=None,
                         encoding='utf-8', engine='python')
        df.columns = columns_map[set_type]
        return df

    def write_raw_data(self, filename, df: pd.DataFrame):
        """以原始格式写入"""
        filepath = self.data_dir / filename
        lines = [[str(w) for w in line] for line in df.values.tolist()]
        lines = ['###__###'.join(line) for line in lines]
        text = '\n'.join(lines)
        filepath.write_text(text, encoding='utf-8')

    def read_all_raw_data(self):
        df_train = self.read_raw_data('train.csv', 'train')
        df_test = self.read_raw_data('text.csv', 'test')
        return pd.concat([df_train, df_test], sort=True, axis=0)

    def write_char_counter(self, filename, counter):
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join([f'{k},{v}' for k, v in counter]))

    def write_list(self, filename, lst):
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lst))

    def read_list(self, filename):
        filepath = self.data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        df = pd.Series(lines, name=os.path.basename(filename))
        return df

    def write_fasttest_input(self, filename, text: pd.Series, label: pd.Series):
        """保存fasttext格式文件"""
        assert len(text) == len(label)
        filepath = self.data_dir / filename
        lst = [f'__label__{l} , {t}' for l, t in zip(list(label), list(text))]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lst))

    def read_fasttext_file(self, fasttext_file):
        """读取fasttext格式文件"""
        filepath = self.data_dir / fasttext_file
        text = filepath.read_text(encoding='utf-8')
        data = [line for line in text.split('\n') if line != '']
        text = [line[13:] for line in data]
        label = [int(i[9]) for i in data]
        return text, label

    def df_to_csv(self, filename, df, *args, **kwargs):
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False, *args, **kwargs)

    def read_csv(self, filename, *args, **kwargs):
        filepath = self.data_dir / filename
        return pd.read_csv(filepath, *args, **kwargs)

    def change_dir(self, data_dir: Path):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            data_dir.mkdir()
        self.data_dir = data_dir


if __name__ == '__main__':
    dataset = DataSet(r'G:\ML\datasets\sougou_personas')
    dataset.read_raw_data('train.csv', 'train')
