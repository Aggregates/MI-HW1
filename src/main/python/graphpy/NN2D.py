import pickle
from numpy import arange,meshgrid,zeros
from matplotlib import pyplot as plot

def plotNN(filePath, lowerBound, upperBound, step):
	
	assert lowerBound <= upperBound

	X = arange(lowerBound, upperBound, step)
	Y = arange(lowerBound, upperBound, step)
	X,Y = meshgrid(X,Y)
	Z = zeros(X.shape)

	model = pickle.load(open(filePath))

	for i in range(len(X)):
	    for j in range(len(Y)):
	        
	        # Classify the result
	        result = model.activate([X[i][j],Y[i][j]])
	        if result[0] > result[1]:
	            Z[i][j] = 0 #lower limit
	        else:
	            Z[i][j] = 100 #higher limit

	plot.imshow(Z)
	plot.gcf()
	plot.clim()
	plot.title("Neural Network Activation")

	return plot

def classifyNN(classes, dataset, resultSet):
	# Compare each result to the expected value
	errors = [0]*len(classes)
	correct = [0]*len(classes)

	for i in xrange( len(resultSet) ):
		# Compare the max activation values
		sampleList = dataset.getSample(i)[1].tolist()
		maxIndex = resultSet[i].index( max(resultSet[i]) )

		# Check if correct
		if maxIndex == sampleList.index(max(sampleList)):
			correct[maxIndex] += 1
		else:
			errors[maxIndex] += 1

	# Plot the classifications
	width = 0.5
	correctPlot = plot.bar( [i for i in xrange( len(classes) )], correct, width, color='b')
	errorPlot   = plot.bar( [i for i in xrange( len(classes) )], errors,  width, bottom=correct, color='r')

	plot.ylabel("# Classifications")
	plot.xticks( [i+width/2 for i in xrange( len(classes) )], classes)

	return plot

def plotBarComparison(errors, correct):
	plot.plot(errors,'r')
	plot.plot(correct,'b')
	plot.ylabel("Training Error")
	plot.xlabel("Training Steps")
	return plot