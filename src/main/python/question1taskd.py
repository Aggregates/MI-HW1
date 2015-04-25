# COMP3330 - Machine Intelligence
# Homework Assignment 1
# Question 1 - Variations of the Two-Spiral Task
# Part d) - Compare the use of ANNs with SVMs to solve tasks a-c

# Imports

import sys
import os

path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path)

from svm.svm import *
from svm.svmutil import svm_train, svm_predict, svm_save_model
from support import csv
#from graphpy import SVM2D # importing this doesn't work for some reason
from time import strftime
from matplotlib import pyplot as plot
from numpy import arange, meshgrid, zeros

KERNEL = RBF # The kernel in the .dll file to use
svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

def plotSVM(solver, lowerBound, upperBound, step):
    
    assert lowerBound < upperBound

    X = arange(lowerBound, upperBound, step)
    Y = arange(lowerBound, upperBound, step)
    X,Y = meshgrid(X,Y)
    Z = zeros(X.shape)

    for i in range(len(X)):
        for j in range(len(Y)):
            
            #Classify the result
            result = int( svm_predict([0.], [[ X[i][j], Y[i][j] ]], solver, '-q')[0][0] )
            if result == 0:
                Z[i][j] = 0 #lower limit
            else:
                Z[i][j] = 100 #higher limit

    plot.imshow(Z)
    plot.gcf()
    plot.clim()
    plot.title("SVM Activation")

    return plot

def taska():
    # Training Parameters
    
    # Defines how high the cost is of a misclassification
    # versus making the decision plane more complex.
    # Low COST makes decisions very simple but creates classification errors
    COST = 0.65

    # Used for generalisation
    # - Low GAMMA means high generalisation
    # - High GAMMA is closer to original dataset
    GAMMA = 3.5

    # Get the data
    dataset, data, outputs, classes = csv.loadSVMProblem("spirals\\SpiralOut.txt")
    
    # Set the parameters for the SVM
    parameters = svm_parameter()
    parameters.kernel_type = KERNEL
    parameters.C = COST
    parameters.gamma = GAMMA

    # Train the SVM
    solver = svm_train(dataset, parameters)

    uniqueFileName = "generated\\Q1DTaskA-TrainedSVM-" + strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
    svm_save_model(uniqueFileName,solver)

    # Compare the results to the extected values
    plot = plotSVM(solver, -6.0, 6.0, 0.2)
    plot.show()

if __name__ == "__main__":
    
    taska()
