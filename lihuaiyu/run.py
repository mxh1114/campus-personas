# -*- coding:utf-8 -*-

from lihuaiyu.model import TextClassifier, XGBoost
from lihuaiyu.model_utils import generate_train_data, to_json
from lihuaiyu.data_utils import read_df_from_csv


def multinomialNB_model_train():
    file = './data/preprocessed.csv'
    group_name = ['Age', 'Gender', 'Education']
    k_fold_attr_name = ['age_kfold_index', 'gender_kfold_index', 'education_kfold_index']
    train_df = read_df_from_csv(file)
    all_group_score = {}
    for i in range(3):
        scores = []
        for j in range(5):
            x_train, y_train, x_test, y_test = generate_train_data(train_df,
                                                                   group_name=group_name[i],
                                                                   k_fold_attr_name=k_fold_attr_name[i],
                                                                   n=j)
            text_classifier = TextClassifier()
            text_classifier.fit(x_train, y_train)
            score = text_classifier.score(x_test, y_test)
            scores.append(score)
            text_classifier.save_model(f"./model/{group_name[i]}_fold_{j}_model.h5")
            print(f'{group_name[i]}- fold-{j} - score: {score}')
        print(scores)
        all_group_score[group_name[i]] = scores
        to_json('./data/score.json', all_group_score)


def generate_mulNb_feature(group_name, model_path):
    model = TextClassifier()
    model.load_model(model_path)
    file = './data/preprocessed.csv'
    df = read_df_from_csv(file)
    train_df = df[df[group_name] != 0]
    Querys = train_df.Query.values
    pre_prob = model.predict_prob(Querys)
    print(pre_prob[:10])


if __name__ == '__main__':
    # multinomialNB_model_train()
    #
    generate_mulNb_feature("Age", './model/age_fold_0_model.h5')
