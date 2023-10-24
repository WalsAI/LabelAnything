import os
import sys
import logging

from Preprocessors.CsvPreprocessor import CsvPreprocessor
from Preprocessors.FolderTransformer import FolderTransformer


class Preprocess:
    def __init__(self, path_to_folder, column, output_csv, logger):
        self.__path_to_folder = path_to_folder
        self.__column = column
        self.__output_csv = output_csv
        self.__logger = logger
        self.__transformer = FolderTransformer(path_to_folder, column, output_csv)
        self.__csv_preprocessor = CsvPreprocessor(output_csv, column)

    def __call__(self):
        self.__logger.info('Preprocessing started...')
        self.__transformer()
        self.__logger.info('All image paths have been written into a csv.')
        self.__logger.info('Preprocessing of the csv file started...')
        self.__csv_preprocessor()
        self.__logger.info('Preprocessing ended.')
