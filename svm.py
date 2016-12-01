# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
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

ligne = X[0,0,:,:,0].shape[0]
col = X[0,0,:,:,0].shape[1]
buffer = np.zeros((ligne*col,1))

for i in range(4):
    x_middle = X[0,i,:,:,0]
    x_middle = np.reshape(x_middle,(-1,1))
    buffer = np.column_stack((buffer,x_middle))
    
X_trainset = buffer[:,(1,2,3,4)]
Y_trainset = np.reshape(Y[0,:,:,0],(-1,1))



clf = svm.SVC()
clf.fit(X_trainset,Y_trainset) 

buffer = np.zeros((ligne*col,1))
for i in range(4):
    x_middle = X[1,i,:,:,0]
    x_middle = np.reshape(x_middle,(-1,1))
    buffer = np.column_stack((buffer,x_middle))
    
X_testset = buffer[:,(1,2,3,4)]
Y_testset = np.reshape(Y[1,:,:,0],(-1,1))
Y_testset = Y_testset.reshape(-1)

y_pred = clf.predict(X_testset)

print jaccard_similarity_score(Y_testset, y_pred)
