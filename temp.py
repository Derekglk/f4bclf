# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
mnist = fetch_mldata('MNIST original', data_home='/homes/xliang/f3b/ACP')

data = mnist.data
label = mnist.target
total = 70000

pca = PCA(n_components=10)
pca.fit(data)
matrix_reduit = pca.transform(data)

image_new = pca.inverse_transform(matrix_reduit)

image0 = data[8000,:]
image0 = np.reshape(image0,(28,28))
plt.subplot(1,2,1)
plt.imshow(image0,cmap ='Greys')

image1 = image_new[8000,:]
image1 = np.reshape(image1,(28,28))
plt.subplot(1,2,2)
plt.imshow(image1,cmap ='Greys')
"""
x = range(625)
plt.subplot(1,2,1)
plt.plot(x, pca.explained_variance_)
plt.subplot(1,2,2)
plt.plot(x, pca.explained_variance_ratio_)
"""

#without PCA
from sklearn import svm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(mnist.data)
data_uni = scaler.transform(mnist.data)

svmrbf = svm.SVC(kernel = 'linear',C=1,gamma = 1,decision_function_shape = 'ovr')
index = np.random.permutation(len(label))
n = 10000
x = data_uni[index[0:n],:]
y = label[index[0:n]]
svmrbf = svmrbf.fit(x,y)

label_predict = svmrbf.predict(data_uni[index[n:total],:])

from sklearn.metrics import accuracy_score
accuracy_score(label[index[n:total]], label_predict)


#SVM
scaler_ = StandardScaler(matrix_reduit)
matrix_reduit = scaler.fit(matrix_reduit)





