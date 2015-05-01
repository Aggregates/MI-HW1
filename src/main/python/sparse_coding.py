import sys
from random import randint
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import one_to_n
from svm.svm import svm_problem

def emptyVector(length):
    emptyCodedVector = []
    for i in xrange(0, length, 1):
        emptyCodedVector.append(0)
    return emptyCodedVector

def codedSample(emptyVector, value):
    sample = list(emptyVector)
    sample[value] = 1
    return sample

def generate(sampleCount, lowerBound=0, upperBound=15):

    dataset = []
    emptyCodedVector = emptyVector(upperBound)

    for i in xrange(sampleCount):
        rand = randint(lowerBound, upperBound-1) # Inclusive
        sample = codedSample(emptyCodedVector, rand)
        dataset.append(sample)

    return dataset

def generateFull(length):
    dataset = []
    emptyCodedVector = emptyVector(length)

    for i in xrange(length):
        sample = codedSample(emptyCodedVector, i)
        dataset.append(sample)
    return dataset


def getClassifier(sample):
    classifier = 0
    for i in range(len(sample)):
        if sample[i] == 1:
            classifier = i
            break
    return classifier


def toClassificationDataset(codedSampleSet):
   
    classifiedSampleSet = []
    
    # Calculate the unique classes
    classes = []
    for sample in codedSampleSet:
    
        classifier = getClassifier(sample)
        if classifier not in classes:
            classes.append(classifier)
    classes.sort()
    
    # Now that we have all the classes, we process the outputs
    for sample in codedSampleSet:
        classifier = getClassifier(sample)
        classifiedSample = one_to_n(classes.index(classifier), len(classes))
        classifiedSampleSet.append(classifiedSample)

    # Build the dataset
    sampleSize = len(codedSampleSet[0])
    classifiedSampleSize = len(classifiedSampleSet[0])
    dataset = ClassificationDataSet(sampleSize, classifiedSampleSize)
    
    for i in range(len(classifiedSampleSet)):
        dataset.addSample(codedSampleSet[i], classifiedSampleSet[i])

    return dataset, classes

def toSVMProblem(codedSampleSet):
    # Calculate the unique classes
    classes = []
    for sample in codedSampleSet:
    
        classifier = getClassifier(sample)
        if classifier not in classes:
            classes.append(classifier)
    classes.sort()

    # Use libsvm's data container:
    return svm_problem([classes.index(i) for i in classes], codedSampleSet), codedSampleSet, codedSampleSet, classes



if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        # Only the filename was input
        print "Error: At least one argument is required"
        print "Expected usage: 'sparse-coding.py sampleCount [lowerBound], [upperBound]'"
    elif len(sys.argv) == 2:
        # One argument - Samples to generate
        print generate( int(sys.argv[1]) )
    elif len(sys.argv) == 4:
        # Too many arguments
        sampleCount = int(sys.argv[1])
        lowerBound  = int(sys.argv[2])
        upperBound  = int(sys.argv[3])
        print generate(sampleCount, lowerBound, upperBound)
    else:
        # Invalid number of arguments
        print "Error: Unexpected number of arguments"
        print "Expected usage: 'sparse-coding.py sampleCount [lowerBound], [upperBound]'"