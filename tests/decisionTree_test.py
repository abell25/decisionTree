import unittest


from decisionTree import DecisionTree
from dataLoader import DataLoader

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.decisionTree = DecisionTree()

    def test_generatesCorrectPrediction(self):
        dataLoader = DataLoader('data/animals.csv')
        X, y = dataLoader.trainingData, dataLoader.labels
        decisionTree = DecisionTree()
        decisionTree.train(X, y)

        result = decisionTree.predict({"Number of legs": 4, "Color": "Yellow"})
        self.assertEqual(result, "Lion")
        
    def test_generatesCorrectPrediction2(self):
        X = [
        {'a': 1, 'b': 9}, {'a': 2, 'b': 5}, {'a': 3, 'b': 7},
        {'a': 4, 'b': 4}, {'a': 5, 'b': 8}, {'a': 6, 'b': 2},
        {'a': 7, 'b': 6}, {'a': 8, 'b': 3}, {'a': 9, 'b': 1}]
        y = [1, 1, 0, 1, 1, 0, 0, 0, 0]
        decisionTree = DecisionTree()
        decisionTree.train(X, y)
        result = [decisionTree.predict_single(x) for x in X]
        self.assertListEqual(result, y)

   