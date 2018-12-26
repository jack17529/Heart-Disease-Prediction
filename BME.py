#This code performs the classification of heart disease by seperatinbg the predicted values
# in two sets, namely 0 for absence and 1 for presence where all the predicted values
#between 1 and 4 are replaced to 1 to check the model performance.

from numpy import genformtxt
import numpy as np
import matpltlib
matplotlib.use('TKAgg',warn=False)
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from itertools import cycle
from sklearn import cross_validation
from sklearn.svm import SVC

#Loading and pruning the data
dataset = genfromtxt('cleveland_data.csv',dtype = float, delimiter=',')
#print dataset
x = dataset[:,0:12] #Feature Set
y = dataset[:,13] #LabelSet

#Replacing 1-4 by 1 label
for index, item in enumerate(y):
    if not(item==0.0):
        y[index]=1
print(y)
target_names = ['0','1']

#Method to plot graph from reduced Dimensions
def plot2D(data,target,target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    plt.figure()
