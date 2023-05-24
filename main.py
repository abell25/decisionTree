from decisionTree import DecisionTree
from dataLoader import DataLoader

if __name__ == '__main__':
    dataLoader = DataLoader('data/animals.csv')
    X, y = dataLoader.trainingData, dataLoader.labels
    decisionTree = DecisionTree()
    decisionTree.train(X, y)

    print('DECISION TREE:')
    print(decisionTree.node.print())

    print('DECISION TREE PREDICTION FOR 4 LEGS & COLOR YELLOW:')
    result = decisionTree.predict({"Number of legs": 4, "Color": "Yellow"})
    print('expected: Lion, actual: {}'.format(result))