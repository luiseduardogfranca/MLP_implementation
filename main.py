#-*- coding utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from data_process import cleanData
import pandas as pd

#defining the activation function
def sig(x, deriv=False):
	if deriv == True:
		return sig(x)*(1-sig(x))

	return 1/(1+np.exp(-x))

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

#Number of neurons on input layer
l0_len = x.shape[-1]
#Neurons on hidden layer
l1_len = 60
#Neurons on output layer
l2_len = 2

eta = 0.00003

#defining weigths random
#l0_len x l1_len
w1 = 2*np.random.random((l0_len ,l1_len))-1

#l1_len x l2_len
w2 = 2*np.random.random((l1_len,l2_len))-1

errors = []

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

	l1_delta = eta*(l0.T.dot(l1_error))
	l2_delta = eta*(l1.T.dot(l2_error))

	w1 += l1_delta
	w2 += l2_delta

df1 = pd.DataFrame(data=w1)
df2 = pd.DataFrame(data=w2)
result = pd.DataFrame(data=errors)
result.to_csv('/output/result.csv', sep=',', encoding='utf-8', index=False)
df1.to_csv('/output/w1.csv',sep=',',encoding='utf-8',index=False)
df2.to_csv('/output/w2.csv',sep=',',encoding='utf-8',index=False)
