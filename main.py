#-*- coding utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_process import train_features, train_labels

def sigmoid(x, deriv=False):
	if deriv:
		return sigmoid(x)*(1-sigmoid(x))
	#1/1+e^-x
	return 1/(1+np.exp(-x))

#Number of neurons on input layer
l0_len = train_features.shape[-1]
#Neurons on hidden layer
l1_len = 50
#Neurons on output layer
l2_len = 2

#learning rate
eta = 0.55

np.random.seed(1)
#defining weigths random
#l0_len x l1_len
# w1 = 2*np.random.random((l0_len, l1_len))-1
w1 = np.random.uniform(-1,1,(l0_len, l1_len))

#l1_len x l2_len
# w2 = 2*np.random.random((l1_len, l2_len))-1
w2 = np.random.uniform(-1,1,(l1_len, l2_len))

# print(np.random.random((l0_len, l1_len)))
errors = []

def train():
	global w1, w2, l0, errors
	for i in range(20000):
		l0 = train_features

		l1 = sigmoid(l0.dot(w1))

		l2 = sigmoid(l1.dot(w2))

		error = train_labels-l2
		# error = ((train_labels - l2)**2)*0.5

		mean_error = np.mean(np.abs(error))
		# print(mean_error)

		if i%100 == 0:
			# print(mean_error)
			errors.append(mean_error)

		l2_error = sigmoid(l2, deriv=True) * error

		l1_error = sigmoid(l1, deriv=True) * (np.dot(l2_error, w2.T))

		l1_delta = eta*(l0.T.dot(l1_error))
		l2_delta = eta*(l1.T.dot(l2_error))

		w1 += l1_delta
		w2 += l2_delta

#
# w1 = pd.DataFrame(data = w1)
# w2 = pd.DataFrame(data = w2)
#
# w1.to_csv('new-w1.csv',sep=',',encoding='utf-8',index=False)
# w1.to_csv('new-w2.csv',sep=',',encoding='utf-8',index=False)

# plt.plot(list(range(len(errors))), errors)
# plt.show()
