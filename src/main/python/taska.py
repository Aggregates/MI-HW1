# COMP3330 - Machine Intelligence
# Homework Assignment 1
# Question 1 - Variations of the Two-Spiral Task
# Part a) - Solve the Two-Spirals Task using a Feedforward Neural Network

# Imports
import sys
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer
import numpy
from matplotlib import pyplot as plot
from support import csv
from graphpy import NN2D

def run():
	# Parameters used for program
	HIDDEN_LAYERS = 5
	WEIGHT_DECAY = 0.01
	MOMENTUM = 0.1
	TRAINING_ITERATIONS = 50
	TRAINING_EPOCHS = 1

	# Import the data for the two spirals Task. Split into 75% training, 25% validation
	dataset = csv.loadRecords("spirals\\SpiralOut.txt")
	validationData, trainingData = dataset.splitWithProportion(0.25)

	# Set up the network and trainer
	inDimension = trainingData.indim
	outDimension = trainingData.outdim
	neuralNet = buildNetwork(inDimension, HIDDEN_LAYERS, outDimension, outclass=SoftmaxLayer)
	trainer = BackpropTrainer(neuralNet, dataset=trainingData, momentum=MOMENTUM, 
		verbose=True, weightdecay=WEIGHT_DECAY)

	# Train the network
	trainingErrors = []
	validationErrors = []

	for i in xrange(TRAINING_ITERATIONS):
		print "Training iteration: ", i

		# The result of training is the proportional error for the number of epochs run
		trainer.trainEpochs(TRAINING_EPOCHS)
		trainingError = trainer.train()
		trainingErrors.append(trainingError)
		#trainingErrorPercentage   = percentError( trainer.trainOnClassData(), trainingData["class"] )
		#validationErrorPercentage = percentError( trainer.trainOnClassData(dataset=validationData), validationData["class"] )
		#trainingErrors.append(trainingErrorPercentage)
		#validationErrors.append(validationErrorPercentage)

		# Activate the network and see what it has learnt
		#activationResult = neuralNet.activateOnDataset(validationData)

		# Display the ressult of training for the iteration
		print "   Epoch: ", trainer.totalepochs, "%"
		print "      Training error:    ", trainingError
		#print "      Training error:    ", trainingErrorPercentage, "%"
		#print "      Validation Error:  ", validationErrorPercentage, "%"
		#print "      Activation Result: ", activationResult, "%"

	# Finished training. Plot the errors
	plot = NN2D.plotBarComparison(trainingErrors, validationErrors)
	plot.show()

# Define ability to run from command line
if __name__ == "__main__":
	run()