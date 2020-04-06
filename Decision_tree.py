#!/usr/bin/env python3

# Henry's work
# Please do not modify it
# Last modify: 6th Apr, 2020

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

np.random.seed(1)
TRAINING_FILE = "training.csv"
TEST_FILE = "test.csv"

def preprocess(df):
    pass

def split_data(df):
    pass

def main():
    df = pd.read_csv(TRAINING_FILE)
    print(df.shape[0], df.shape[1])


if __name__ == "__main__":
    main()