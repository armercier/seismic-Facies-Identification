import numpy as np
from data import *

def importData(dataTrain,labelsTrain,dataTest):
    data_train = np.load(dataTrain, allow_pickle=True, mmap_mode='r')
    data_train = data_train['data']

    labels_train = np.load(labelsTrain, allow_pickle=True, mmap_mode='r')
    labels_train = labels_train['labels']

    data_test = np.load(dataTest, allow_pickle=True, mmap_mode='r')
    data_test = data_test['data']

    labels_train -= 1
    return data_train, labels_train, data_test

def normalizeData(dataTrain,dataTest):
    adjusted_test_img = np.concatenate([dataTrain, dataTest], axis=2)

    _min, _max = adjusted_test_img.min(), adjusted_test_img.max()

    train = (dataTrain - _min) / (_max - _min)
    test = (dataTrain - _min) / (_max - _min)
    adjusted_test_img = (adjusted_test_img - _min) / (_max - _min)

    del adjusted_test_img
    return train, test



