from typing import Optional, Dict, List, Union
from csv import DictReader

class DataLoader(object):
    def __init__(
            self,
            filename: str,
            trainStringColumns: List[str] = ["Color"],
            trainIntColumns: List[str] = ["Number of legs"],
            labelColumn: str = "Name"):
        self.filename = filename
        self.data = self.loadData(filename)
        for row in self.data:
            for column in trainIntColumns:
                row[column] = int(row[column])
        trainColumns = trainStringColumns + trainIntColumns
        self.trainingData = [{k:v for k, v in row.items() if k in trainColumns} for row in self.data]
        self.labels = [row[labelColumn] for row in self.data]


    def loadData(self, filename: str) -> List[Dict[str, Union[int, str]]]:
        with open(filename, 'r') as csvfile:
            reader = DictReader(csvfile, delimiter=',', quotechar='"')
            data = list(reader)
            return data