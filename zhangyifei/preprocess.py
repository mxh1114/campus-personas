# -*- coding: utf-8 -*-
import re
from collections import Counter

import jieba
from sklearn.model_selection import KFold

from zhangyifei.config import *
from zhangyifei.data_utils import *

TASK_LIST = ['age', 'gender', 'education']
dataset = DataSet(data_dir)


def get_uniq_char(df):
    """字符统计"""
    counter = Counter()
    for text in df['query'].values.tolist():
        counter.update(Counter(text))
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    dataset.write_char_counter('char_counter.txt', counter)


def word_tokenize(filename):
    """分词，结果保存到`data_dir`/filename，每行1个样本，单词之间空格分隔"""
    df = dataset.read_raw_data('train.csv', 'train')
    tokenized_words = []
    for ind, text in enumerate(df['query'].values.tolist()):
        words = []
        for query in text.split('\t'):
            tokenized_query = [w if w != ' ' else '<BLANK>'
                               for w in jieba.lcut(query)]
            tokenized_query += ['<EOS>']
            words.extend(tokenized_query)
        tokenized_words.append(' '.join(words))
        if ind % 10000 == 0:
            print(f'processing {ind}')
    dataset.write_list(filename, tokenized_words)


def generate_cv_dataset(n_splits=5):
    """生成交叉验证分组数据集"""
    tokenized_text = dataset.read_list('train_clean_tokenized.txt')
    labels = dataset.read_csv('train_clean_labels.csv')
    dataset.change_dir(Path(data_dir, 'cross_validation'))

    sk = KFold(n_splits=n_splits, shuffle=True, random_state=2020)
    cv_groups = sk.split(tokenized_text)
    for ind, (train_idx, valid_idx) in enumerate(cv_groups):
        for task in TASK_LIST:
            label_train = labels.loc[train_idx, task]
            label_valid = labels.loc[valid_idx, task]
            dataset.df_to_csv(f'{task}_train_cv{ind}.csv', label_train)
            dataset.df_to_csv(f'{task}_valid_cv{ind}.csv', label_valid)

        text_train = tokenized_text[train_idx]
        text_valid = tokenized_text[valid_idx]
        dataset.write_list(f'text_train_cv{ind}.csv', text_train)
        dataset.write_list(f'text_valid_cv{ind}.csv', text_valid)


def generate_fasttext_input():
    dataset.change_dir(Path(data_dir, 'cross_validation'))
    fasttext_dataset = DataSet(Path(data_dir, 'fasttext'))
    for ind in range(5):
        text_train = dataset.read_list(f'text_train_cv{ind}.csv')
        text_valid = dataset.read_list(f'text_valid_cv{ind}.csv')
        for task in TASK_LIST:
            label_train = dataset.read_list(f'{task}_train_cv{ind}.csv')
            label_valid = dataset.read_list(f'{task}_valid_cv{ind}.csv')
            fasttext_dataset.write_fasttest_input(f'{task}_train_cv{ind}.txt',
                                                  text_train, label_train)
            fasttext_dataset.write_fasttest_input(f'{task}_valid_cv{ind}.txt',
                                                  text_valid, label_valid)


def get_vocab():
    lines = dataset.read_list('train_tokenized.txt')
    counter = Counter()
    for line in lines:
        words = line.split()
        counter.update(Counter(words))
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    dataset.write_char_counter('vocab.txt', counter)

    # 不是中文的词
    pattern = re.compile(r'[^\u4e00-\u9fff]+')
    non_chinese_word = [(w, c) for w, c in counter if pattern.search(w)]
    dataset.write_char_counter('non_cn_vocab.txt', non_chinese_word)

    # 不是数字、标点、英文的词
    pattern = re.compile(
        r'[^\d#/A-Za-z=()_\-+:,.\[\]!·?，。、！\'"《》—【】~|]+')
    special_word = [(w, c) for w, c in non_chinese_word if pattern.search(w)]
    dataset.write_char_counter('special_word.txt', special_word)


def clean():
    """
    数据清洗
    1. 删除age, gender, education有缺失的样本
    2. 待定
    :return:
    """
    df_train = dataset.read_raw_data('train.csv', 'train')

    # 删除缺失数据
    df_train = df_train.query('age > 0 and gender > 0 and education > 0').copy()
    for label in ['age', 'gender', 'education']:
        df_train[label] -= 1
    df_labels = df_train[['age', 'gender', 'education']]
    dataset.df_to_csv('train_clean_labels.csv', df_labels)

    train_tokenized = dataset.read_list('train_tokenized.txt')
    train_tokenized = train_tokenized[df_train.index]
    dataset.write_list('train_clean_tokenized.txt', train_tokenized)


def stat_word_frequency():
    """3个任务不同类别分别统计词频，计算词频的标准差"""

    df = dataset.read_csv('train_clean_labels.csv')
    train_tokenized = dataset.read_list('train_clean_tokenized.txt')

    for label in ['age', 'gender', 'education']:
        grp = train_tokenized.groupby(by=df[label])
        word_frequency = {}
        for name, series in grp:
            counter = Counter()
            for text in series:
                counter.update(Counter(text.split()))
            word_frequency[name] = counter
        df_freq = pd.DataFrame(word_frequency)
        df_freq.fillna(0, inplace=True)
        # 因样本是不均衡的，除以各类别人数
        df_freq = df_freq / df[label].value_counts()
        df_freq.columns = [f'{label}_{i}' for i in df_freq.columns]
        df_freq['std'] = df_freq.std(axis=1)
        df_freq.sort_values(by='std', ascending=False, inplace=True)
        df_freq.index.name = 'word'
        df_freq.reset_index(drop=False, inplace=True)
        dataset.df_to_csv(f'{label}_freq_std.csv', df_freq)


def substitute():
    """一个或多个连续空格 -> <BLANK>, '\t' -> <EOS>"""
    sub_list = [
        (re.compile(r" +"), '<BLANK>'),
        (re.compile(r"\t"), '<EOS>')
    ]

    def _sub(text):
        for pattern, string_ in sub_list:
            text = re.sub(pattern, string_, text)
        return text

    df = dataset.read_raw_data('train.csv', 'train')
    df['query'] = df['query'].apply(_sub)
    dataset.write_raw_data('train_sub.csv', df)


def main():
    # 分词
    word_tokenize('train_tokenized.txt')

    # 暂且先删除了缺失数据，可考虑补值。
    clean()

    stat_word_frequency()

    # 5折交叉验证分组
    generate_cv_dataset(n_splits=5)

    # 生成fasttext输入文件
    generate_fasttext_input()


if __name__ == '__main__':
    main()
