import numpy as np
from data import *

def importData(dataTrain,labelsTrain,dataTest):
    data_train = np.load(dataTrain, allow_pickle=True, mmap_mode='r')
    data_train = data_train['data']

    labels_train = np.load(labelsTrain, allow_pickle=True, mmap_mode='r')
    labels_train = labels_train['labels']

    data_test = np.load(dataTest, allow_pickle=True, mmap_mode='r')
    data_test = data_test['data']
    return data_train, labels_train, data_test




