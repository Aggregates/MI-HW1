# COMP3330 - Machine Intelligence
# Homework Assignment 1
# Question 1 - Variations of the Two-Spiral Task
# Part a) - Solve the Two-Spirals Task using a Feedforward Neural Network

# Imports
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from matplotlib import pyplot as plot
from support import csv
from graphpy import NN2D
from time import strftime

import sys
import numpy
import pickle

def run():
    # Parameters used for program
    HIDDEN_LAYERS = 5
    LEARNING_DECAY = 0.990 # Set in range [0.9, 1]
    LEARNING_RATE = 0.0005 # Set in range [0, 1]
    MOMENTUM = 0.1 # Set in range [0, 0.5]
    TRAINING_ITERATIONS = 10
    BATCH_LEARNING = False
    VALIDATION_PROPORTION = 0.0

    # Import the data for the two spirals Task
    dataset, classes = csv.loadCSV("spirals\\SpiralOut.txt")

    # Set up the network and trainer
    inDimension = dataset.indim
    outDimension = dataset.outdim
    neuralNet = buildNetwork(inDimension, HIDDEN_LAYERS, outDimension)
    trainer = BackpropTrainer(neuralNet, dataset, learningrate=LEARNING_RATE, momentum=MOMENTUM, 
    	lrdecay=LEARNING_DECAY, batchlearning=BATCH_LEARNING)

    # Train the network
    trainingErrors = []
    validationErrors = []

    for i in xrange(TRAINING_ITERATIONS):
        print "Training iteration: ", i

        # Check if VALIDATION_PROPORTION is not 0. This will split the input dataset into
        # VALIDATION_PROPORTION % for Validation Data and
        # (1 - VALIDATION_PROPORTION) % for Training Data
        # e.g. 25% ValidationData and 75% Training Data

        if VALIDATION_PROPORTION == 0.0 or VALIDATION_PROPORTION == 0:
            # Cannot split the data set into Training and Validation Data. Train the 
            # Neural Network by standard means. This will not calculate Validatinon Error

            # The result of training is the proportional error for the number of epochs run
            trainingError = trainer.train()
            trainingErrors.append(trainingError)
        else:
            trainingErrors, validationErrors = trainer.trainUntilConvergence(validationProportion=VALIDATION_PROPORTION)

        # Display the result of training for the iteration
        print "   Training error:    ", trainingError

    # Save the Trained Neural Network
    uniqueFileName = "generated\\TaskA-TrainedNN-" + strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
    writeMode = 'wb' # Write Bytes
    pickle.dump(neuralNet, open(uniqueFileName, writeMode))

    # Plot the results of training
    plot = NN2D.plotNN(network=neuralNet, lowerBound=-6.0, upperBound=6.0, step=0.2)
    plot.show()

    if VALIDATION_PROPORTION != 0.0 or VALIDATION_PROPORTION != 0:
        plot.clf() # Clear figure?
        plot = NN2D.plotBarComparison(trainingErrors, validationErrors)
        plot.show()


# Define ability to run from command line
if __name__ == "__main__":
    run()