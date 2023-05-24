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
        

   