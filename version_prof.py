# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:36:28 2016

@author: jshi
"""

import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import jaccard_similarity_score


fx = gzip.open('/homes/jshi/S5/F4B_305A/Classification_TP1/x.pklz','rb')
X = pickle.load(fx)
fx.close()
fy = gzip.open('/homes/jshi/S5/F4B_305A/Classification_TP1/y.pklz','rb')
Y = pickle.load(fy)
fy.close()

x = np.transpose(X[1,:,:,:,:].reshape((4,-1)))
y = np.transpose(Y[1,:,:,:].reshape((-1)))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn import svm
svmrbf = svm.SVC(kernel = 'rbf',C=1,gamma = 1,decision_function_shape = 'ovr')
svmrbf = svmrbf.fit(x,y)

y_pred = svmrbf.predict(x)
y_pred = y_pred.reshape(240,240,10)

plt.imshow(y_pred[:,:,1])
