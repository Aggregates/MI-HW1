# COMP3330 - Machine Intelligence
# Homework Assignment 1
# Question 2 - Autoencoder
# Train a 16-H-16 multilayer perceptron to map a number to its vector form

# Imports
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from graphpy import NN2D
from time import strftime
from pybrain.tools.shortcuts import buildNetwork
from os import path, makedirs
from svm.svm import *
from svm.svmutil import svm_train, svm_predict, svm_save_model
from time import strftime, time
from mpl_toolkits.mplot3d import Axes3D

import sys
import numpy
import pickle
import sparse_coding
import matplotlib.pyplot as plot

def part1():
    '''
    Determine the minimal number of hidden units
    required to train the network successfully
    '''
    
    # Parameters
    HIDDEN_NODES =          [8]
    LEARNING_DECAY =        0.999501    # Set in range [0.9, 1]
    LEARNING_RATE =         0.324501    # Set in range [0, 1]
    MOMENTUM =              0.101    # Set in range [0, 0.5]
    TRAINING_ITERATIONS =   5000
    BATCH_LEARNING =        False
    VALIDATION_PROPORTION = 0.0
    SPARSE_LENGTH =         16

    # Get the dataset
    dataset = sparse_coding.generateFull(SPARSE_LENGTH)
    validationSet = sparse_coding.generateFull(SPARSE_LENGTH)
    dataset, classes = sparse_coding.toClassificationDataset(dataset)
    inDimension = dataset.indim
    outDimension = dataset.outdim

    print inDimension
    print outDimension

    layers = [inDimension] + HIDDEN_NODES + [outDimension]
    neuralNet = buildNetwork(*layers)
    
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


    # Create the output path if it doesn't exist
    generated_dir = path.abspath(path.join("generated", "Q2Task1-TrainedNN-{}".format(strftime("%Y-%m-%d_%H-%M-%S"))))
    if not path.exists(generated_dir):
        makedirs(generated_dir)

    # save parameters
    with open(path.normpath(path.join(generated_dir, "params.txt")), "a") as f:
      f.write("HIDDEN_LAYERS = {}\n".format(HIDDEN_NODES))
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


    # Plot the results of training
    plot.plot(trainingErrors, 'b')
    plot.ylabel("Training Error")
    plot.xlabel("Training Steps")
    plot.savefig(path.normpath(path.join(generated_dir, "errors.png")))
    plot.show()
    plot.clf()
    
    figure = plot.figure()
    axis = figure.add_subplot(111, projection='3d')
    colors = ['r','y','g','c','b','k']

    for sample in validationSet:
        classifier = sparse_coding.getClassifier(sample)
        activationResult = neuralNet.activate(sample)
        axis.bar(range(len(sample)), activationResult, classifier, zdir='y', color=colors[:len(sample)])

    plot.savefig(path.normpath(path.join(generated_dir, "activations.png")))
    plot.show()


def part2():
    '''
    Determine the minimal number of hidden units
    required to train the network successfully
    using multiple hidden layers
    '''

    '''
    # Parameters
    HIDDEN_NODES =          8
    LEARNING_DECAY =        0.9999    # Set in range [0.9, 1]
    LEARNING_RATE =         0.08    # Set in range [0, 1]
    MOMENTUM =              0.0    # Set in range [0, 0.5]
    TRAINING_ITERATIONS =   1000
    BATCH_LEARNING =        False
    VALIDATION_PROPORTION = 0.0
    SPARSE_LENGTH =         16
    '''


    # Parameters
    HIDDEN_NODES =          4
    LEARNING_DECAY =        0.9999    # Set in range [0.9, 1]
    LEARNING_RATE =         0.111    # Set in range [0, 1]
    MOMENTUM =              0.05    # Set in range [0, 0.5]
    TRAINING_ITERATIONS =   5000
    BATCH_LEARNING =        False
    VALIDATION_PROPORTION = 0.0
    SPARSE_LENGTH =         16

    # Get the dataset
    dataset = sparse_coding.generateFull(SPARSE_LENGTH)
    validationSet = sparse_coding.generateFull(SPARSE_LENGTH)
    dataset, classes = sparse_coding.toClassificationDataset(dataset)
    inDimension = dataset.indim
    outDimension = dataset.outdim

    print inDimension
    print outDimension

    # Set up the neral network layers
    inLayer = LinearLayer(inDimension, name='input')
    hiddenLayer1 = SigmoidLayer(HIDDEN_NODES, name='hidden1')
    hiddenLayer2 = TanhLayer(HIDDEN_NODES, name='hidden2')
    outLayer = LinearLayer(outDimension, name='output')

    # Set up the connections
    input_to_hidden1 = FullConnection(inLayer, hiddenLayer1, name='in_h1')
    hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2, name='h1_h2')
    hidden2_to_output = FullConnection(hiddenLayer2, outLayer, name='h2_out')
    hidden1_to_output = FullConnection(hiddenLayer1, outLayer, name='h2_out')

    # Create the network and add the information
    neuralNet = FeedForwardNetwork()
    neuralNet.addInputModule(inLayer)
    neuralNet.addModule(hiddenLayer1)
    neuralNet.addModule(hiddenLayer2)
    neuralNet.addOutputModule(outLayer)

    neuralNet.addConnection(input_to_hidden1)
    neuralNet.addConnection(hidden1_to_hidden2)
    neuralNet.addConnection(hidden2_to_output)
    neuralNet.addConnection(hidden1_to_output)
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


    # Create the output path if it doesn't exist
    generated_dir = path.abspath(path.join("generated", "Q2Task2-TrainedNN-{}".format(strftime("%Y-%m-%d_%H-%M-%S"))))
    if not path.exists(generated_dir):
        makedirs(generated_dir)

    # save parameters
    with open(path.normpath(path.join(generated_dir, "params.txt")), "a") as f:
      f.write("HIDDEN_LAYERS = {}\n".format(HIDDEN_NODES))
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


    # Plot the results of training
    plot.plot(trainingErrors, 'b')
    plot.ylabel("Training Error")
    plot.xlabel("Training Steps")
    plot.savefig(path.normpath(path.join(generated_dir, "errors.png")))
    plot.show()
    plot.clf()
    
    from mpl_toolkits.mplot3d import Axes3D
    figure = plot.figure()
    axis = figure.add_subplot(111, projection='3d')
    colors = ['r','y','g','c','b','k']

    for sample in validationSet:
        classifier = sparse_coding.getClassifier(sample)
        activationResult = neuralNet.activate(sample)
        axis.bar(range(len(sample)), activationResult, classifier, zdir='y', color=colors[:len(sample)])

    plot.savefig(path.normpath(path.join(generated_dir, "activations.png")))
    plot.show()

def svm():
    # Training Parameters
    
    # Defines how high the cost is of a misclassification
    # versus making the decision plane more complex.
    # Low COST makes decisions very simple but creates classification errors
    COST = 0.9

    # Used for generalisation
    # - Low GAMMA means high generalisation
    # - High GAMMA is closer to original dataset
    GAMMA = 6

    KERNEL = RBF
    svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

    # Get the data
    SPARSE_LENGTH = 16
    sparseCodings = sparse_coding.generateFull(SPARSE_LENGTH)
    dataset, data, outputs, classes = sparse_coding.toSVMProblem(sparseCodings)
    
    # Set the parameters for the SVM
    parameters = svm_parameter()
    parameters.kernel_type = KERNEL
    parameters.C = COST
    parameters.gamma = GAMMA

    # Train the SVM
    solver = svm_train(dataset, parameters)

    # Create the output path if it doesn't exist
    generated_dir = path.abspath(path.join("generated", "Q2Task1-TrainedSVM-{}".format(strftime("%Y-%m-%d_%H-%M-%S"))))
    if not path.exists(generated_dir):
        makedirs(generated_dir)

    uniqueFileName = path.normpath(path.join(generated_dir, "data.pkl"))
    svm_save_model(uniqueFileName,solver)
    
    # Compare the results to the extected values
    figure = plot.figure()
    axis = figure.add_subplot(111)
    colors = ['r','y','g','c','b','k']

    for sample in sparseCodings:
        classifier = sparse_coding.getClassifier(sample)
        activationResult = svm_predict([0.], [sample], solver, '-q')[0][0]
        axis.bar(classifier, activationResult, color=colors[classifier % len(colors)])

    plot.savefig(path.normpath(path.join(generated_dir, "activations.png")))
    plot.show()

if __name__ == "__main__":
    part2()