# COMP3330 - Machine Intelligence
# Homework Assignment 1
# Question 1 - Variations of the Two-Spiral Task
# Part a) - Solve the Two-Spirals Task using a Feedforward Neural Network

# Imports
import sys
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
import numpy
from matplotlib import pyplot as plot
from support import csv

def hello():
	print "Hello From Q1"

def run():
	# Import the data for the two spirals Task
	dataset, classes = csv.loadCSV("spirals\\spiralsdataset.txt")
	print dataset

if __name__ == "__main__":
	run()