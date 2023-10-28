import pandas as pd
from typing import List, Any


class CsvPreprocessor:
    def __init__(self, file: str, column: str) -> None:
        """
            Class for preprocessing the csv and converting the rows from column into a list
        :param file: csv path
        :param column: column that you are interested in
        """
        self.__file = file
        self.__column = column

    def csv_to_list(self) -> List[Any]:
        """
            Copies all the elements from that column into a list
        :return: list containing the specific elements
        """
        images = pd.read_csv(self.__file)
        images = images[self.__column].tolist()

        return images

    def __call__(self) -> List[Any]:
        """
            Make the class callable
        :return: List[Any]
        """
        return self.csv_to_list()
