from utils import *
# import matplotlib
# matplotlib.use('Qt5Agg') # MUST BE CALLED BEFORE IMPORTING plt
import matplotlib.pyplot as plt

train, labels, test = importData('data/data_train.npz','data/labels_train.npz','data/data_test_1.npz')

plt.imshow(labels[500,:,:])
plt.show()
