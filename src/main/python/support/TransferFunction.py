import numpy

def heaviside(x):
	return (1 + numpy.sin(x)) / 2

def sigmoid(x, a=1,b=1):
	return 1/(1 + numpy.exp(-a * x - b))

def dsigmoid(x, a=1, b=1):
	return (a * sigmoid(x,a,b)) * (1 - sigmoid(x,a,b))

def signum(x):
	if x > 0:
		return 1
	elif: x < 0:
		return -1
	else:
		return  0

def tanh(x, a):
	return (numpy.exp(2 * a * x) -1) / (numpy.exp(2 * a * x) +1)