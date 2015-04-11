from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import pyplot as plot
import numpy

def visualise(X,Y,neuralNet):
	grid = X.copy() * 0.0
	#Iterate through coordinates in X,Y space
	for x in xrange(len(X)):
		for y in xrange(len( X[0] )):
			grid[x][y] = neuralNet.activate( [ X[x][y], Y[x][y] ] )
	return grid

def NNSurfaceHyperplane(neuralNet, lowerBound, upperBound, step):
	xvalues = numpy.arange(lowerBound, upperBound, step)
	yvalues = numpy.arange(lowerBound, upperBound, step)
	X,Y = numpy.meshgrid(xvalues, yvalues)
	grid = visualise(X, Y, neuralNet)

	figure = plot.figure()
	axis = figure.add_subplot(111, projection='3d')
	surface = axis.plot_surface(X,Y,grid, 
		rstride=1, cstride=1, cmap=cm.hsv,
		linewidth=0, antialiased=False)
	figure.colorbar(surface, shrink=0.5, aspect=5)

	return figure