import pandas as pd


class CsvPreprocessor:
    def __init__(self, file, column):
        self.__file = file
        self.__column = column

    def csv_to_list(self):
        images = pd.read_csv(self.__file)
        images = images[self.__column].tolist()

        return images

    def __call__(self):
        return self.csv_to_list()
