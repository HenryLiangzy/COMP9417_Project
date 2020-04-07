#!/usr/bin/env python3

'''
Henry's work
Please DO NOT modify this file
Currently for testing
Last modify: 6th Apr, 2020
'''

import time
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

np.random.seed(1)
TRAINING_FILE = "training.csv"
TEST_FILE = "test.csv"

topic_code = {
    'ARTS CULTURE ENTERTAINMENT': 1,
    'BIOGRAPHIES PERSONALITIES PEOPLE': 2,
    'DEFENCE': 3,
    'DOMESTIC MARKETS': 4,
    'FOREX MARKETS': 5,
    'HEALTH': 6,
    'MONEY MARKETS': 7,
    'SCIENCE AND TECHNOLOGY': 8,
    'SHARE LISTINGS': 9,
    'SPORTS': 10,
    'IRRELEVANT': 0
}

def preprocess(df):
    df['topic_code'] = df['topic'].apply(lambda x: topic_code[x])

    return df[['article_words', 'topic_code']]

def bag_of_word(df, model):
    vector = model
    df[model.__class__.__name__] = df['article_words'].apply(lambda x: vector.fit_transform(x))

    return df

def main():

    # load data from file
    df = pd.read_csv(TRAINING_FILE)

    # pre process the y
    df = preprocess(df)

    # split the data
    train_set, test_set = train_test_split(df, test_size=0.1)

    # print(train_set)

    count = CountVectorizer()

    train_x = count.fit_transform(train_set['article_words'].values)
    train_y = train_set['topic_code']
    test_x = count.transform(test_set['article_words'].values)
    test_y = test_set['topic_code']

    print(train_x.shape)
    print(train_y.shape[0])
    print(test_x.shape)
    print(test_y.shape[0])
    print("Start training")
    start_time = time.time()

    # # train
    # test_record = []
    # trianing_record = []

    dtc = DecisionTreeClassifier()
    dtc.fit(train_x, train_y)

    end_time = time.time()

    print("The accuracy of Training set is:", dtc.score(train_x, train_y))
    print("The accuracy of Test set is:", dtc.score(test_x, test_y))
    print("Training time: {} s".format(end_time-start_time))

    # Part B: search min_samples_leaf between 2~20
    # train_min_samples_leaf_value = 0
    # train_max_auc = 0

    # min_samples_leaf_value = 0
    # max_auc = 0
    # leaf_value_range = range(2, 21)
    # for leaf_value in leaf_value_range:
    #     model = DecisionTreeClassifier(min_samples_leaf=leaf_value)
    #     model.fit(train_x, train_y)

    #     prediction = model.predict_proba(test_x)
    #     t_prediction = model.predict_proba(train_x)
    #     fpr, tpr, thresholds = roc_curve(test_x, prediction[:,1])
    #     t_fpr, t_tpr, t_thresholds = roc_curve(train_y, t_prediction[:,1])

    #     auc = metrics.auc(fpr, tpr)
    #     t_auc = metrics.auc(t_fpr, t_tpr)
    #     test_record.append(auc)
    #     trianing_record.append(t_auc)

    #     if auc > max_auc:
    #         max_auc = auc
    #         min_samples_leaf_value = leaf_value
    #     if t_auc > train_max_auc:
    #         train_max_auc = t_auc
    #         train_min_samples_leaf_value = leaf_value

    # print("The optimal number of min_samples_leaf by TEST set is:", min_samples_leaf_value)
    # print("With max AUC by TEST is:", max_auc)
    # print("The optimal number of min_samples_leaf by TRAINING set is:", train_min_samples_leaf_value)
    # print("With max AUC by TRAINING is:", train_max_auc)


    # plt.style.use('ggplot')
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(121)
    # ax1.plot(leaf_value_range, test_record)
    # ax1.set_title('AUC in TEST set')
    # ax1.set_xlabel('iterations')
    # ax1.set_ylabel('AUC')

    # ax2 = fig.add_subplot(122)
    # ax2.plot(leaf_value_range, trianing_record)
    # ax2.set_title('AUC in TRAINING set')
    # ax2.set_xlabel('iterations')
    # ax2.set_ylabel('AUC')
    
    # plt.show()
    # #plt.savefig('figure/Assignment2_Q2_part_C.png')
    

if __name__ == "__main__":
    main()