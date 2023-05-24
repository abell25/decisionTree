import unittest

from dataLoader import DataLoader

class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_loadData(self):
        dataLoader = DataLoader('data/animals.csv')
        data = dataLoader.data
        self.assertEqual(len(data), 5)
        self.assertListEqual(
            [x["Name"] for x in data], ["Lion", "Monkey", "Parrot", "Snake", "Bear"])
        self.assertListEqual(
            [x["Number of legs"] for x in data], [4, 4, 2, 0, 4])
        self.assertListEqual(
            [x["Color"] for x in data], ["Yellow", "Black", "Green", "Green", "Black"])

    def test_trainingData(self):
        dataLoader = DataLoader('data/animals.csv')
        trainingData = dataLoader.trainingData 
        self.assertEqual(len(trainingData), 5)
        self.assertSetEqual(set(trainingData[0]), set(["Number of legs", "Color"]))
        labels = dataLoader.labels
        self.assertEqual(len(labels), 5)
        self.assertEqual(labels, ["Lion", "Monkey", "Parrot", "Snake", "Bear"])