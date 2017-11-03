# from main import w1, w2, percent, x_test, y_test
from data_process import cleanData
import numpy as np
from main import train_cross_val
from activation_functions import LeakyReLU, sigmoid

def start():

	#10% of the all data
	percent_test = 0.1

	n_loops = len(cleanData)//int(percent_test*len(cleanData))
	steps = int(percent_test*len(cleanData))

	scores_total = []

	#test method
	def test(x,y, w1, w2):
		predictions = []
		scores = 0

		for i in range(len(x)):
			l0 = x[i]
			l1 = LeakyReLU(l0.dot(w1))
			l2 = sigmoid(l1.dot(w2))

			predictions.append([1,0]) if l2[0] > l2[-1] else predictions.append([0,1])
			if predictions[i][0] == y[i][0] and predictions[i][1] == y[i][1]:
				scores += 1

		return scores/len(y)


	features = cleanData[:, :-1]
	labels = cleanData[:, -1]

	bias = np.ones(len(features))
	features = np.c_[features, bias]

	temp = []
	for i in labels:
		temp.append([1,0]) if i == 2 else temp.append([0,1])

	labels = np.array(temp)

	#cross validation
	j = 0

	print("\nPercent of test: %.2f%% \nAmount cases for test: %d\n" %(percent_test * 100, steps))

	for i in range(n_loops):

		features_cross = features
		labels_cross = labels

		test_features = features_cross[j:(i+1)*steps, :]

		features_cross = np.delete(features_cross, np.s_[j:(i+1)*steps], 0)

		train_features = features_cross

		test_labels = labels_cross[j:(i+1)*steps, :]

		labels_cross = np.delete(labels_cross, np.s_[j:(i+1)*steps], 0)

		train_labels = labels_cross

		w1, w2 = train_cross_val(train_features, train_labels)

		scores_total.append(test(test_features, test_labels, w1, w2))

		print("%dº validation: \t%.2f%%" %((i + 1), 100 * scores_total[i]))

		j = (i+1)*steps

	print("\nTotal accurancy: %0.3f%%"%(np.mean(scores_total)*100))
