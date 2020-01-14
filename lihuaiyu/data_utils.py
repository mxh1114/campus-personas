# -*- coding:utf-8 -*-
import os
import re
import jieba
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import jieba

tqdm.pandas(desc='apply')

# 加载停用词
stopwords = pd.read_csv('./data/stopwords.txt', sep='\t', header=None, names=['stopword'],
                        quoting=3, encoding='utf-8', engine='python')
STOPWORD = set(stopwords.stopword.values)


def genrate_k_fold_index(category, group: dict, n=5):
    """
 生成每条数据的5-fold 的索引index
    :param category:
    :param group:
    :param n: n-flod number
    :return:
    """
    index = group[category] % n
    group[category] += 1
    return index


def matcher(string):
    '''匹配非中文字符串'''
    pattern = re.compile('[^\u4e00-\u9fff ]+')
    return set(re.findall(pattern, string))


def cut_sentences(sentences):
    """ 切词 ，默认连续非中文字符串为一个词 不包括空格"""
    splits = sentences.split("\t")
    total_words = []
    for sen in splits:
        match_result = matcher(sen)
        if match_result:
            for w in match_result:
                if len(w) == 1:
                    continue
                sen = sen.replace(w, f'\t->{w}\t')
            words = []
            for sub_sen in sen.split('\t'):
                if sub_sen.startswith('->'):
                    words.append(sub_sen[2:])
                    continue
                words.extend(jieba.lcut(sub_sen))
            #             print(words)
            total_words.append('\t'.join(words))
            continue
        total_words.append('\t'.join(jieba.lcut(sen)))
    return '\t\t'.join(total_words)


def df_to_csv(file, df):
    df.to_csv(file, sep=str(","), header=None, index=False, encoding='utf-8')
    print('saved ...')


def read_df_from_csv(file, names=None):
    # 重新载入数据
    if names is None:
        names = ['ID', 'Age', 'Gender', 'Education', 'age_kfold_index', 'gender_kfold_index', 'education_kfold_index',
                 'query_num', 'query_max_len', 'query_ave_len', 'query_min_len', 'blank_rate', 'english_rate', 'Query']
    data_df = pd.read_csv(file, sep=',', header=None, names=names)
    return data_df


def is_contain_letter(str0):
    import re
    return bool(re.search('[a-zA-Z]', str0))


def query_stat(querys):
    # 提取query特征
    """

    :param querys:
    :return:
    """
    query_splits = querys.split('\t\t')
    # Query的数量
    query_num = len(query_splits)
    # Query的平均长度
    query_ave_length = 0
    # Query的最大长度
    query_max_length = 0
    # Query的最小长度
    query_min_length = 1000
    # 空格率
    blank_rate = 0
    # 字母率
    english_rate = 0

    for single_query in query_splits:
        single_query = single_query.replace('\t', '')
        query_length = len(single_query)
        query_max_length = query_length if query_length > query_max_length else query_max_length
        query_min_length = query_length if query_length < query_min_length else query_min_length
        query_ave_length += query_length
        if " " in single_query:
            blank_rate += 1
        if is_contain_letter(single_query):
            english_rate += 1
    #         print(query_length)

    query_ave_length /= query_num
    blank_rate /= query_num
    english_rate /= query_num
    stat_list = [query_num, query_max_length, query_ave_length, query_min_length, blank_rate, english_rate]

    return stat_list


def _process(querys):
    ''' 去停用词'''
    words = []
    for single_query in querys.split('\t\t'):
        for word in single_query.split('\t'):
            if word in STOPWORD:
                continue
            words.append(word)
    return ' '.join(words)


# 数据读取和词分割、生成k-fold index
def preprocess(file):
    # 初始化每一子类样本总数量词典
    age_subclass_dict = defaultdict(int)
    gender_subclass_dict = defaultdict(int)
    education_subclass_dict = defaultdict(int)

    names = ['ID', 'Age', 'Gender', 'Education', 'Query']

    data_dtype = {'ID': np.str, "Age": np.int16, 'Gender': np.int16, 'Education': np.int16, 'Query': np.str}
    df = pd.read_csv(file, sep='###__###', header=None, names=names, dtype=data_dtype, encoding='utf-8',
                     engine='python')

    df['age_kfold_index'] = df['Age'].progress_apply(lambda x: genrate_k_fold_index(x, age_subclass_dict))
    df['gender_kfold_index'] = df['Gender'].progress_apply(lambda x: genrate_k_fold_index(x, gender_subclass_dict))
    df['education_kfold_index'] = df['Education'].progress_apply(
        lambda x: genrate_k_fold_index(x, education_subclass_dict))
    # 分词
    df['Query'] = df['Query'].progress_apply(lambda x: cut_sentences(x))
    # 特征生成
    df['query_stat'] = df['Query'].progress_apply(lambda x: query_stat(x))
    df['query_num'] = df['query_stat'].progress_apply(lambda x: x[0])
    df['query_max_len'] = df['query_stat'].progress_apply(lambda x: x[1])
    df['query_ave_len'] = df['query_stat'].progress_apply(lambda x: x[2])
    df['query_min_len'] = df['query_stat'].progress_apply(lambda x: x[3])
    df['blank_rate'] = df['query_stat'].progress_apply(lambda x: x[4])
    df['english_rate'] = df['query_stat'].progress_apply(lambda x: x[5])
    # Query去停用词,且空格化
    df['Query'] = df['Query'].progress_apply(lambda x: _process(x))
    names = ['ID', 'Age', 'Gender', 'Education', 'age_kfold_index', 'gender_kfold_index', 'education_kfold_index',
             'query_num', 'query_max_len', 'query_ave_len', 'query_min_len', 'blank_rate', 'english_rate', 'Query']
    df = df[names]
    return df


if __name__ == '__main__':
    file = './data/train.csv'
    preprocess_df = preprocess(file)
    preprocess_file = './data/preprocessed.csv'
    df_to_csv(preprocess_file, preprocess_df)
