from Preprocessors.FolderTransformer import FolderTransformer
from logging import Logger


class Preprocess:
    def __init__(self, path_to_folder: str, column: str, output_csv: str, logger: Logger) -> None:
        """
            Class for preprocessing the folder
        :param path_to_folder: path to folder in which the images are stored
        :param column: name of the csv column to store the image paths
        :param output_csv: name of the output csv
        :param logger: logger used to log the warnings and info
        """
        self.__path_to_folder = path_to_folder
        self.__column = column
        self.__output_csv = output_csv
        self.__logger = logger
        self.__transformer = FolderTransformer(path_to_folder, column, output_csv)

    def __call__(self) -> None:
        """
            Make the class callable
        :return: None
        """
        self.__logger.info('Preprocessing started...')
        self.__transformer()
        self.__logger.info('All image paths have been written into a csv.')
        self.__logger.info('Preprocessing ended.')
