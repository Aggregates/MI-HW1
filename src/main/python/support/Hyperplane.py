import numpy, scipy

__author__ = 'Tyler 2'


class Hyperplane:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.hyperplaneData = numpy.random.randn(dimensions+1)

    def evaluate(self, values):
        values = numpy.concatenate([values, numpy.array([1])])
        return numpy.dot(self.hyperplaneData, values)

    def neuron(self, values):
        return 1.0 / (1.0 + numpy.exp(numpy.dot(self.hyperplaneData, values)))