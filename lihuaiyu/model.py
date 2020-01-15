# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from joblib import dump, load
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid


class TextClassifier():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def predict_prob(self, x):
        return self.classifier.predict_proba(self.features(x))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

    def save_model(self, path):
        dump((self.classifier, self.vectorizer), path)

    def load_model(self, path):
        self.classifier, self.vectorizer = load(path)


class XGBoost(object):
    def __init__(self):
        self.model = self._create_model()

    def _create_model(self):
        return XGBClassifier(booster='gbtree',
                             learning_rate=0.1,
                             n_estimators=600,  # 树的个数--1000棵树建立xgboost
                             max_depth=8,  # 树的深度
                             min_child_weight=1,  # 叶子节点最小权重
                             max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
                             gamma=0.,  # 惩罚项中叶子结点个数前的参数
                             subsample=0.8,  # 随机选择80%样本建立决策树
                             colsample_btree=0.8,  # 随机选择80%特征建立决策树
                             # objective='multi:softmax',  # 指定损失函数
                             objective='binary:logistic',
                             # scale_pos_weight=0.25,  # 解决样本个数不平衡的问题
                             random_state=1000,  # 随机数,
                             # num_class=2,
                             reg_alpha=0,
                             reg_lambda=5,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                             silent=True
                             )

    def fit(self, X, y, eval_set,
            eval_metric='logloss',
            early_stopping_rounds=50, verbose=None):
        self.model.fit(X, y, eval_set=eval_set,
                       eval_metric=eval_metric,
                       early_stopping_rounds=early_stopping_rounds,
                       verbose=verbose)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def save_model(self, path):
        dump(self.model, path)
        print(f'{path} 已保存!!! ')

    def load_model(self, path):
        self.model = load(path)

    def plot_importance_feature(self, keys=None):
        fig, ax = plt.subplots(figsize=(15, 15))

        if keys is None:
            plot_importance(self.model, height=0.5, ax=ax, max_num_features=20, )
        else:
            scores = self.model.get_booster().get_score(importance_type='weight')
            score_dict = {}
            print(scores)
            for i in range(len(keys)):
                f = f'f{i}'
                if f in scores:
                    score_dict[keys[i]] = scores[f]
            plot_importance(score_dict, height=0.5, ax=ax, max_num_features=20, )

        plt.show()

    def search(self, X, Y, params=None):
        if params is None:
            params = {
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [4, 6, 8, 10]
            }
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        gsearch = GridSearchCV(estimator=self.model, return_train_score=True,
                               param_grid=params,
                               scoring='roc_auc',
                               n_jobs=2,
                               iid=False,
                               cv=kfold,
                               verbose=2
                               )
        gsearch.fit(X, Y)
        print(gsearch.param_grid)
        print(gsearch.best_score_)
        print(gsearch.best_params_)
        print(gsearch.best_estimator_)
        print(gsearch.best_index_)
        # print(gsearch.cv_results_)
        mean_train_score = gsearch.cv_results_['mean_train_score']
        std_train_score = gsearch.cv_results_['std_train_score']
        mean_test_score = gsearch.cv_results_['mean_test_score']
        std_test_score = gsearch.cv_results_['std_test_score']
        params = gsearch.cv_results_['params']
        for x, y, z, w, t in zip(params, mean_train_score, std_train_score, mean_test_score, std_test_score):
            print(f'params: {x}, mean_train_score:{y}, std_train_score:{z}, mean_test_score:{w}, std_test_score:{t}')

        return gsearch.param_grid, gsearch.best_score_, gsearch.best_params_

    def select_n_estimators(self, X, Y, cv_folds=5, early_stopping_rounds=50):
        xgb_param = self.model.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=self.model.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=1)
        print(cvresult.shape[0])
        self.model.set_params(n_estimators=cvresult.shape[0])
