#!/usr/bin/env python2.7
# COMP3330 - Machine Intelligence
# Homework Assignment 1
# Question 1 - Variations of the Two-Spiral Task
# Part c)

# Imports
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, MDLSTMLayer, TanhLayer, LSTMLayer
from pybrain.structure import FullConnection
from pybrain.tools.shortcuts import buildNetwork
from support import csv
from graphpy import NN2D
from time import strftime
from os import path, makedirs

import sys
import numpy
import pickle

def run():
    # Parameters used for program
    HIDDEN_LAYERS = map(lambda _: 55, range(0,3))
    LEARNING_DECAY = 0.999999 # Set in range [0.9, 1]
    LEARNING_RATE = 0.1 # Set in range [0, 1]
    MOMENTUM = 0 # Set in range [0, 0.5]
    TRAINING_ITERATIONS = 15000
    BATCH_LEARNING = False
    VALIDATION_PROPORTION = 0.0

    # Import the data for the two spirals Task
    dataset, classes = csv.loadCSV(path.abspath('spirals/4Spirals.txt'))

    # Set up the network and trainer
    inDimension = dataset.indim
    outDimension = dataset.outdim

    layers = [inDimension] + HIDDEN_LAYERS + [outDimension]
    neuralNet = buildNetwork(*layers)

    print neuralNet

    trainer = BackpropTrainer(neuralNet, dataset, learningrate=LEARNING_RATE, momentum=MOMENTUM, 
    	lrdecay=LEARNING_DECAY, batchlearning=BATCH_LEARNING)

    # Train the network
    trainingErrors = []
    validationErrors = []

    for i in xrange(TRAINING_ITERATIONS):
        print "Training iteration: ", i
        if trainingErrors < 0.001:
            break

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

            # Display the result of training for the iteration
            print "   Training error:    ", trainingError
        else:
            trainingErrors, validationErrors = trainer.trainUntilConvergence(validationProportion=VALIDATION_PROPORTION)

    # create path if it doesn't exist
    generated_dir = path.abspath(path.join("generated", "Q1TaskC-TrainedNN-{}".format(strftime("%Y-%m-%d_%H-%M-%S"))))
    if not path.exists(generated_dir):
      makedirs(generated_dir)

    # save parameters
    with open(path.normpath(path.join(generated_dir, "params.txt")), "a") as f:
      f.write("HIDDEN_LAYERS = {}\n".format(HIDDEN_LAYERS))
      f.write("LEARNING_DECAY = {}\n".format(LEARNING_DECAY))
      f.write("LEARNING_RATE = {}\n".format(LEARNING_RATE))
      f.write("MOMENTUM = {}\n".format(MOMENTUM))
      f.write("TRAINING_ITERATIONS = {}\n".format(TRAINING_ITERATIONS))
      f.write("BATCH_LEARNING = {}\n".format(BATCH_LEARNING))
      f.write("VALIDATION_PROPORTION = {}\n".format(VALIDATION_PROPORTION))

    # Save the Trained Neural Network
    uniqueFileName = path.normpath(path.join(generated_dir, "data.pkl"))

    writeMode = 'wb' # Write Bytes
    pickle.dump(neuralNet, open(uniqueFileName, writeMode))

    import matplotlib.pyplot as plot

    # Plot the results of training
    plot.plot(trainingErrors, 'b')
    plot.ylabel("Training Error")
    plot.xlabel("Training Steps")
    plot.savefig(path.normpath(path.join(generated_dir, "errors.png")))
    plot.show()
    plot.clf()
    
    plot = NN2D.plotNN(network=neuralNet, lowerBound=-6.0, upperBound=6.0, step=0.1)

    if VALIDATION_PROPORTION != 0.0 or VALIDATION_PROPORTION != 0:
      plot = NN2D.plotBarComparison(trainingErrors, validationErrors)

    plot.savefig(path.normpath(path.join(generated_dir, "result.png")))
    plot.show()


# Define ability to run from command line
if __name__ == "__main__":
    run()