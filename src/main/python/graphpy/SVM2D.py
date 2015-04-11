from svmutil import svm_load_model, svm_predict
from svm import *
from numpy import arange, meshgrid, zeros
from matplotlib import pyplot as plot
from numpy import array

def plotSVM(filePath, lowerBound, upperBound, step):
	
	assert lowerBound < upperBound

	X = arange(lowerBound, upperBound, step)
	Y = arange(lowerBound, upperBound, step)
	X,Y = meshgrid(X,Y)
	Z = zeros(X.shape)

	model = svm_load_model('mysvm')

	for i in range(len(X)):
	    for j in range(len(Y)):
	        
	        #Classify the result
	        result = int( svm_predict([0.], [[ X[i][j], Y[i][j] ]], model, '-q')[0][0] )
	        if result == 0:
	            Z[i][j] = 0 #lower limit
	        else:
	            Z[i][j] = 100 #higher limit

	plot.imshow(Z)
	plot.gcf()
	plot.clim()
	plot.title("SVM Activation")

	return plot

def classifySVM(classes, data, outputs, solver):
	errors = [0]*len(classes)
	correct = [0]*len(classes)

	for i in xrange( len(data) ):
		answer = int( solver.predict(data[i]) )
		print answer
		if answer != outputs[i]:
			errors[answer] += 1
		else:
			correct[answer] += 1

	width = 0.5
	correctPlot = plot.bar( [i for i in xrange( len(classes) ) ], correct, width, color='b')
	errorPlot   = plot.bar( [i for i in xrange( len(classes) ) ], errors,  width, bottom=correct, color='r')

	plot.ylabel("# Classifications")
	plot.xticks( [i+width/2 for i in xrange( len(classes) )], classes)
	
	return plot