#-*- coding utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import webview
from data_process import train_features, train_labels

def sigmoid(x, deriv=False):
	if deriv:
		return sigmoid(x)*(1-sigmoid(x))
	#1/1+e^-x
	return 1/(1+np.exp(-x))

def LeakyReLU(x, deriv=False):
	if deriv == True:
		return 1. * (x >= 0) + (0.01*(x<0))
	return ((x < 0 )*(0.01*x)) + ((x >= 0)*x)


np.random.seed(1)

#Number of neurons on input layer + bias(1)
l0_len = train_features.shape[-1]
#Neurons on hidden layer
l1_len = 50
#Neurons on output layer
l2_len = 2

#learning rate
eta = 0.00003

np.random.seed(1)

#defining weigths random
#l0_len x l1_len
w1 = np.random.uniform(-1,1,(l0_len, l1_len))

#l1_len x l2_len
w2 = np.random.uniform(-1,1,(l1_len, l2_len))

errors = []

def train(amountRepetition = 20000):
	global w1, w2, errors

	errors = []

	# 1% and 10%
	percent_one = amountRepetition * 0.01
	percent_ten = amountRepetition * 0.1

	count_one_percent = 0
	count_ten_percent = 0

	char_load = ("#" * count_ten_percent)
	char_blankspace = (" " * (10 - count_ten_percent))

	for i in range(1, amountRepetition + 1):
		#l0 -> Input layer
		l0 = train_features

		#l1 -> Hidde layer output
		l1 = LeakyReLU(l0.dot(w1))

		#l2 -> Output layer out
		l2 = sigmoid(l1.dot(w2))

		error = train_labels-l2

		mean_error = np.mean(np.abs(error))

		if i%100 == 0:
			errors.append(mean_error)

		l2_error = sigmoid(l2, deriv=True) * error

		l1_error = LeakyReLU(l1, deriv=True) * (np.dot(l2_error, w2.T))

		l1_delta = eta*(l0.T.dot(l1_error))
		l2_delta = eta*(l1.T.dot(l2_error))

		w1 += l1_delta
		w2 += l2_delta

		#progress bar
		if(i % percent_one == 0): #while percent is equal to 1%
			count_one_percent += 1

			if(i % percent_ten == 0): #while percent is equal to 10%
				count_ten_percent += 1
				char_load = ("#" * count_ten_percent)
				char_blankspace = (" " * (10 - count_ten_percent))

			loading = ("[%s%s]%d%%" %(char_load, char_blankspace, count_one_percent))
			sys.stdout.write(loading)
			sys.stdout.flush()

			if(count_one_percent != 100):
			    sys.stdout.write("\b" * (13 + len(str(i))))

def train_cross_val(train_features, train_labels, amountRepetition = 1000):
	global w1, w2
	for i in range(amountRepetition):
		l0 = train_features

		l1 = LeakyReLU(l0.dot(w1))

		l2 = sigmoid(l1.dot(w2))

		error = train_labels-l2

		mean_error = np.mean(np.abs(error))

		l2_error = sigmoid(l2, deriv=True) * error

		l1_error = LeakyReLU(l1, deriv=True) * (np.dot(l2_error, w2.T))

		l1_delta = eta*(l0.T.dot(l1_error))
		l2_delta = eta*(l1.T.dot(l2_error))

		w1 += l1_delta
		w2 += l2_delta

	return w1, w2

def saveSVG(fileName1, fileName2):
	global w1, w2

	df1 = pd.DataFrame(data=w1)
	df2 = pd.DataFrame(data=w2)

	df1.to_csv('weights/' + fileName1,sep=',',encoding='utf-8',index=False)
	df2.to_csv('weights/' + fileName2,sep=',',encoding='utf-8',index=False)

def printGraph():
	global plt
	plt.plot(list(range(len(errors))), errors)
	plt.show()

def neuralNetworkSimulation():
	webview.create_window("Neural Network","https://luiseduardogfranca.github.io/neural-network/neural-network.html")
