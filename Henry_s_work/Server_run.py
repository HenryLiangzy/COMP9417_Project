import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.base import TransformerMixin
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline, Pipeline
import joblib
import matplotlib.pyplot as plt


np.random.seed(1)

TRAININGFILE = '../keyword.csv'
TESTFILE = '../key_word_test.csv'
TESTSIZE = 0.1

x_label_list = ['article_words','key_word_50', 'key_word_100']
y_label_list = ['topic']

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

def preprocess(df, x_label, y_label, split=False):
    '''
    Return the x and y columns for trainning
    '''

    if split == True:
        train_set, test_set = train_test_split(df, test_size=TESTSIZE)
        return train_set, test_set
    else:
        return df[[x_label, y_label]]


# for the bag of word and label encode process
def convert_word(bag_of_word_model, label_model, data_set, x_label, y_label='topic'):
    '''
    bow model need to be pre-fit when call current function
    '''
    act_x = bag_of_word_model.transform(data_set[x_label].values)
    act_y = label_model.transform(data_set[y_label])

    return act_x, act_y


def smote_with_vector(df, vector_model, label_model, x_label):
    '''
    df                      data set
    vector_model            Bag of Word model
    x_label                 process x column
    y_label                 process y column
    '''
    
    count = vector_model.fit(df[x_label])

    # convert the data
    train_x, train_y = convert_word(count, label_model, df, x_label)

    # start to SMOTE
    smote = SMOTE(random_state=1)
    sm_x, sm_y = smote.fit_sample(train_x, train_y)

    # re-cover the data
    new_x = count.inverse_transform(sm_x)
    new_x = pd.Series([','.join(item) for item in new_x])

    return new_x, sm_y


def grid_search(vector, model, train_x, train_y):
    kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)

    pipe = Pipeline([
        ('vector', vector),
        ('model', model)
    ])

    param_grid = {
        'model__splitter': ['best', 'random'],
        'model__max_depth': range(10, 100, 10),
        'model__min_samples_split': range(2, 10, 2),
        'model__min_samples_leaf': range(1, 9, 2),
        'model__class_weight': [None, 'balanced']
    }

    # param_grid = {
    #     'model__splitter': ['best']
    # }

    grid_search = GridSearchCV(pipe, param_grid, cv=kfold, n_jobs=-1)
    grid_result=grid_search.fit(train_x, train_y)
    return (grid_result.best_estimator_,grid_result.best_score_)

def topic_score(model, label_model, data_set, topic_name, x_label):
    test_data_set = data_set[data_set['topic'] == topic_name]
    test_x = test_data_set[x_label]
    test_y = test_data_set['topic']
    pred_y = model.predict(test_x)
    en_test_y = label_model.transform(test_y)

    f1_score = metrics.f1_score(en_test_y, pred_y, average='macro')
    accuarcy = metrics.accuracy_score(en_test_y, pred_y)
    recall_score = metrics.recall_score(en_test_y, pred_y, average='macro')

    return {
        'f1': round(f1_score, 4),
        'accuarcy': round(accuarcy, 4),
        'recall_score': round(recall_score, 4)
    }


def model_score(model, label_model, x_label, test_df):
    '''
    model       The dt model
    test_df     provide testing data set or using test file data
    '''
    
    print('Topic\tf1\taccuarcy\trecall_score')
    test_report = []

    test_df = preprocess(test_df, x_label, 'topic')

    for topic in topic_code.keys():
        result = [topic]
        result.append(topic_score(model, label_model, test_df, topic, x_label))
        test_report.append(result)

    test_report.sort(reverse=True, key=lambda x: x[1]['accuarcy'])
    for record in test_report:
        print(record)

    return test_report


def save_job(model, test_report, pre_vector, feature_name):
    filename = 'model/'+str(pre_vector)+'_'+feature_name

    joblib.dump(model, filename+'.model')
    with open(filename+'.txt', 'w') as fp:
        fp.write('Topic\tf1\taccuarcy\trecall_score\n')
        for record in test_report:
            fp.write(str(record)+'\n')


def model_compile(df, x_label, vector_num):

    print('Trainning topic', x_label, 'with vector num', vector_num)
    df = preprocess(df, x_label, 'topic')

    label_model = preprocessing.LabelEncoder().fit(df['topic'])
    encode_mapping = dict(zip(label_model.classes_, range(len(label_model.classes_))))

    if vector_num == 1:
        train_x, train_y = smote_with_vector(df, TfidfVectorizer(), label_model, x_label)
    else:
        train_x, train_y = smote_with_vector(df, CountVectorizer(), label_model, x_label)
    topic = topic_code.keys()

    # prepared for grid-search
    count_dt_model, count_dt_accuarcy = grid_search(CountVectorizer(), DecisionTreeClassifier(), train_x, train_y)
    tfidf_dt_model, tfidf_dt_accuarcy = grid_search(TfidfVectorizer(norm=None), DecisionTreeClassifier(), train_x, train_y)

    if count_dt_accuarcy >= tfidf_dt_accuarcy:
        print(f'*************************************************************')
        print(f'Now the training set is {x_label}, and the model chosen is count_clf_NB')
        print(f'The accuracy is {count_dt_accuarcy}')
        return count_dt_model,label_model,encode_mapping
    else:
        print(f'*************************************************************')
        print(f'Now the training set is {x_label}, and the model chosen is tfidf_clf_NB')
        print(f'The accuracy is {tfidf_dt_accuarcy}')
        return tfidf_dt_model,label_model,encode_mapping

def model_evaluate(model, x_label, label_model, df, encode_mapping, vector_num):
    print('Start to evalute', x_label, 'model')
    test_set = preprocess(df, x_label, 'topic')
    test_x = test_set[x_label]
    test_y = test_set['topic']
    topics = list(set(test_set['topic']))

    # evalute total performance
    pred_y = model.predict(test_x)
    en_test_y = label_model.transform(test_y)
    print('Total proformance')
    print('F1 score:', metrics.f1_score(en_test_y, pred_y, average='macro'))
    print('Accuarcy:', metrics.accuracy_score(en_test_y, pred_y))
    print('Recall score:', metrics.recall_score(en_test_y, pred_y, average='macro'))
    print('-'*15)
    print('Classification Report:')
    print(classification_report(en_test_y, pred_y))

    # evalute all the topic performance
    model_report = model_score(model, label_model, x_label, df)

    # save current model and performance
    save_job(model, model_report, vector_num, x_label)

    # for figure
    conf_matrix = confusion_matrix(en_test_y, pred_y)
    fig1 = plt.figure(figsize=(13,6))
    sns.heatmap(conf_matrix,
    #             square=True,
                annot=True, # show numbers in each cell
                fmt='d', # set number format to integer in each cell
                yticklabels=label_model.classes_,
                xticklabels=model.classes_,
                cmap="Blues",
    #             linecolor="k",
                linewidths=.1,
               )
    plt.title(
              f"Confusion Matrix on Test Set | " 
              f"Classifier: {'+'.join([step for step in model.named_steps.keys()])}", 
              fontsize=14)
    plt.xlabel("Actual: False positives for y != x", fontsize=12)
    plt.ylabel("Prediction: False negatives for x != y", fontsize=12)
    plt.show()
    #plt.savefig('model/'+str(vector_num)+'_'+x_label+'.png')


if __name__ == "__main__":
    # load data
    df = pd.read_csv(TRAININGFILE)
    test_df = pd.read_csv(TESTFILE)
    for x_label in x_label_list:
        for vector_num in [1, 2]:
            model, label_model, encode_mapping = model_compile(df, x_label, vector_num)
            model_evaluate(model, x_label, label_model, test_df, encode_mapping, vector_num)