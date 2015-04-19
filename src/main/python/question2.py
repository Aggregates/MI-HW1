# COMP3330 - Machine Intelligence
# Homework Assignment 1
# Question 2 - Autoencoder
# Train a 16-H-16 multilayer perceptron to map a number to its vector form

# Imports
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from graphpy import NN2D
from time import strftime

import sys
import numpy
import pickle
import sparse_coding

def part1():
    """
    Determine the minimal number of hidden units
    required to train the network successfully
    """

    # Parameters
    HIDDEN_NODES = 3
    LEARNING_DECAY = 0.99999 # Set in range [0.9, 1]
    LEARNING_RATE = 0.30 # Set in range [0, 1]
    MOMENTUM = 0.1 # Set in range [0, 0.5]
    TRAINING_ITERATIONS = 500
    BATCH_LEARNING = False
    VALIDATION_PROPORTION = 0.0
    TRAINING_ITERATIONS = 1

    # Get the dataset
    dataset = sparse_coding.generate(10)
    dataset, classes = sparse_coding.toClassificationDataset(dataset)
    inDimension = dataset.indim
    outDimension = dataset.outdim

    # Set up the neral network layers
    inLayer = LinearLayer(inDimension, name='input')
    hiddenLayer1 = SigmoidLayer(HIDDEN_NODES, name='hidden1')
    hiddenLayer2 = SigmoidLayer(HIDDEN_NODES, name='hidden2')
    outLayer = LinearLayer(outDimension, name='output')

    # Set up the connections
    input_to_hidden1 = FullConnection(inLayer, hiddenLayer1, name='in_h1')
    hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2, name='h1_h2')
    hidden2_to_output = FullConnection(hiddenLayer2, outLayer, name='h2_out')

    # Create the network and add the information
    neuralNet = FeedForwardNetwork()
    neuralNet.addInputModule(inLayer)
    neuralNet.addModule(hiddenLayer1)
    neuralNet.addModule(hiddenLayer2)
    neuralNet.addOutputModule(outLayer)
    neuralNet.addConnection(input_to_hidden1)
    neuralNet.addConnection(hidden1_to_hidden2)
    neuralNet.addConnection(hidden2_to_output)
    neuralNet.sortModules()

    print neuralNet

    # Train the network
    trainer = BackpropTrainer(neuralNet, dataset, learningrate=LEARNING_RATE, momentum=MOMENTUM, 
        lrdecay=LEARNING_DECAY, batchlearning=BATCH_LEARNING)

    trainingErrors = []
    validationErrors = []

    for i in xrange(TRAINING_ITERATIONS):
        print "Training iteration: ", i

        # Check if VALIDATION_PROPORTION is not 0. This will split the input dataset into
        # VALIDATION_PROPORTION % for Validation Data and
        # (1 - VALIDATION_PROPORTION) % for Training Data
        # e.g. 25% Validation Data and 75% Training Data

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

    # Save the Trained Neural Network
    uniqueFileName = "generated\\Q2Task1-TrainedNN-" + strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
    writeMode = 'wb' # Write Bytes
    pickle.dump(neuralNet, open(uniqueFileName, writeMode))


    import matplotlib.pyplot as plot

    # Plot the results of training
    plot.plot(trainingErrors, 'b')
    plot.ylabel("Training Error")
    plot.xlabel("Training Steps")
    plot.show()
    plot.clf()
    
    plot = NN2D.plotNN(network=neuralNet, lowerBound=-6.0, upperBound=6.0, step=0.2)
    plot.show()

    if VALIDATION_PROPORTION != 0.0 or VALIDATION_PROPORTION != 0:
        plot.clf() # Clear figure?
        plot = NN2D.plotBarComparison(trainingErrors, validationErrors)
        plot.show()


if __name__ == "__main__":
    part1()

