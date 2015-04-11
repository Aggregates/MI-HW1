import numpy

def squaredError(response, target):
	return (target - response) ** 2

def sumSquaredError(responses, targets):
	
	assert len(responses) == len(targets)

	result = 0
	for i in xrange( len(responses) ):
		error = squaredError(responses[i], targets[i])
		result += error
	return result

def sumSquaredError(patterns):
	responses = patterns[0]
	targets  = patterns[1]
	return sumSquaredError(responses, targets)

def meanSquaredError(patterns):
	return sumSquaredError(patterns) / len(patterns)

def rootMeanSquaredError(patterns):
	return numpy.sqrt(meanSquaredError)