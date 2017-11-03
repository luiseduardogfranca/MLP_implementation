from main import w1, w2, train, saveSVG, printGraph, neuralNetworkSimulation
import cross_validation


print("\n======== Implemenatation MLP - Neural Network========\n")
print("1 - Train neural network \n2 - Print weight of neural network \n3 - Show graphic of training \n4 - Save weights as \n5 - Neural Network Simulation \n6 - Cross Validation\n0 - Sair \n")

option = input("Option: ")

while (option != "0"):

	if(option == "1"):
		number = int(input("\nNumber of repetitions:"))
		print("\nTraining in progress \n\t")
		train(number)

	elif(option == "2"):
		print("\nWeights of first layer: \n\n", w1, "\n\nWeights of secont layer: \n\n", w2)

	elif(option == "3"):
		printGraph()

	elif(option == "4"):
		nameFirstLayer = input("\nFile name for weights of first layer: ")
		nameSecondLayer = input("File name for weights of second layer: ")

		nameFirstLayer += ".csv"
		nameSecondLayer += ".csv"

		saveSVG(nameFirstLayer, nameSecondLayer)

	elif(option == "5"):
		neuralNetworkSimulation()

	elif(option == "6"):
		cross_validation.start()

	else:
		print("Invalid option")



	print("\n======== Implemenatation MLP - Neural Network========\n")
	print("1 - Train neural network \n2 - Print weight of neural network \n3 - Show graphic of training \n4 - Save weights as \n5 - Neural Network Simulation \n6 - Cross Validation\n0 - Sair \n")

	option = input("Option: ")

print("\n======== Finalized ========")
