import os
import pandas as pd


class FolderTransformer:
    """
        This class transforms a folder into a CSV
    """

    def __init__(self, path_to_folder, column, output_csv):
        self.__path_to_folder = path_to_folder
        self.__column = column
        self.__output_csv = output_csv

    def convert(self):
        images_df = pd.DataFrame(columns=[self.__column])
        list_of_images = list()
        for image in os.listdir(self.__path_to_folder):
            list_of_images.append(self.__path_to_folder + '/' + image)
        images_df = images_df.from_dict({self.__column: list_of_images})

        images_df.to_csv(self.__output_csv, index=False, mode='w')

    def __call__(self):

        self.convert()
