import unittest
from sparse_coding import *

class TestSparseCodingGenerator(unittest.TestCase):

    def text_emptyVector_length5(self):
        vector = emptyVector(5)
        expected = [0, 0, 0, 0, 0]
        self.assertEqual(len(vector), len(expected))

    def test_emptyVector_length10(self):
        vector = emptyVector(10)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert len(vector) == len(expected)

    def test_codedSample_length10_value5(self):
        vector = codedSample(emptyVector(10), 5)
        expected = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        assert len(vector) == len(expected)

        for i in range(len(expected)):
            assert vector[i] == expected[i]

    def test_codedSample_length10_value9(self):
        vector = codedSample(emptyVector(10), 9)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        assert len(vector) == len(expected)

        for i in range(len(expected)):
            assert vector[i] == expected[i]

    def test_Generate_length5(self):
        vector = generate(1, 0, 5)
        expected = [[1,0,0,0,0]]
        assert len(vector) == len(expected)

    def test_Generate_length10(self):
        vector = generate(1, 0, 10)
        expected = [[1,0,0,0,0,0,0,0,0,0]]
        assert len(vector) == len(expected)

    def test_GetClassifier_Length5_Class2(self):
        vector = codedSample(emptyVector(5), 2)
        assert getClassifier(vector) == 2

    def test_GetClassifier_Length10_Class7(self):
        vector = codedSample(emptyVector(10), 7)
        assert getClassifier(vector) == 7

if __name__ == '__main__':
    unittest.main()
