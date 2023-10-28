import os
import pandas as pd


class FolderTransformer:

    def __init__(self, path_to_folder: str, column: str, output_csv: str) -> None:
        """
            Class that writes the paths of images from a folder into a csv for easier access.
        :param path_to_folder: path to the folder in which the images are stored
        :param column: name of the column where the image paths will be stored
        :param output_csv: name of the output csv
        """
        self.__path_to_folder = path_to_folder
        self.__column = column
        self.__output_csv = output_csv

    def convert(self) -> None:
        """
            Does the specific conversion (i.e. folder to csv)
        :return: None
        """
        images_df = pd.DataFrame(columns=[self.__column])
        list_of_images = list()
        for image in os.listdir(self.__path_to_folder):
            list_of_images.append(self.__path_to_folder + '/' + image)
        images_df = images_df.from_dict({self.__column: list_of_images})

        images_df.to_csv(self.__output_csv, index=False, mode='w')

    def __call__(self) -> None:
        """
            Make the class Callable
        :return: None
        """
        self.convert()
