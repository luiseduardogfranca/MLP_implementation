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

<<<<<<< HEAD
def LeakyReLU(x, deriv=False):
	if deriv == True:
		return 1. * (x >= 0) + (0.01*(x<0))
	return ((x < 0 )*(0.01*x)) + ((x >= 0)*x)

#spliting labels and features
x = [i[:-1] for i in cleanData]
y = [i[-1] for i in cleanData]

bias = np.ones(len(x))

x = np.array(x)
y = np.array(y)

temp = []

for i in y:
	temp.append([1,0]) if i == 2 else temp.append([0,1])

y = np.array(temp)

#Adding the bias term
x = np.c_[x, bias]

percent = 0.9

x = np.array([i for i in x[:int(percent*len(x))]])
x_test = np.array([i for i in x[int((1-percent)*len(x)):]])

y = np.array([i for i in y[:int(percent*len(y))]])

np.random.seed(1)

=======
>>>>>>> 476ed8e46a4069205d333111ba2cd92d0e3707ba
#Number of neurons on input layer
l0_len = train_features.shape[-1]
#Neurons on hidden layer
l1_len = 60
#Neurons on output layer
l2_len = 2

<<<<<<< HEAD
eta = 0.00003
=======
#learning rate
eta = 0.55
>>>>>>> 476ed8e46a4069205d333111ba2cd92d0e3707ba

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

<<<<<<< HEAD
i = 1

for i in range(6000):
	l0 = x
	#calculating the hide layer output
	l1 = LeakyReLU(l0.dot(w1))
	#calculating the output layer result
	l2 = sig(l1.dot(w2))
	#abosulte error
	l2_error = y - l2

	mean_error = np.mean(np.abs(l2_error))

	if i%100 == 0:
		print(mean_error)
		errors.append(mean_error)

	l2_error = sig(l2, deriv=True)*l2_error

	l1_error = LeakyReLU(l1, deriv=True)*(np.dot(l2_error,w2.T))
=======
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
>>>>>>> 476ed8e46a4069205d333111ba2cd92d0e3707ba

		l1_delta = eta*(l0.T.dot(l1_error))
		l2_delta = eta*(l1.T.dot(l2_error))

		w1 += l1_delta
		w2 += l2_delta

<<<<<<< HEAD
df1 = pd.DataFrame(data=w1)
df2 = pd.DataFrame(data=w2)
result = pd.DataFrame(data=errors)
result.to_csv('/output/result.csv', sep=',', encoding='utf-8', index=False)
df1.to_csv('/output/w1.csv',sep=',',encoding='utf-8',index=False)
df2.to_csv('/output/w2.csv',sep=',',encoding='utf-8',index=False)
=======
#
# w1 = pd.DataFrame(data = w1)
# w2 = pd.DataFrame(data = w2)
#
# w1.to_csv('new-w1.csv',sep=',',encoding='utf-8',index=False)
# w1.to_csv('new-w2.csv',sep=',',encoding='utf-8',index=False)

# plt.plot(list(range(len(errors))), errors)
# plt.show()
>>>>>>> 476ed8e46a4069205d333111ba2cd92d0e3707ba
