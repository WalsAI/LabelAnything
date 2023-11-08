from typing import Any
from groundingdino.util.inference import load_image, load_model, predict, annotate

import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
sys.path.append('..')


from Preprocessors.CsvPreprocessor import CsvPreprocessor
from Preprocessors.FolderTransformer import FolderTransformer
from utils import utils

class DinoInference:
    def __init__(self, model_config_path: str, model_checkpoint_path: str, label_file:str , csv_file:str, image_column: str, logger: str):
        """Class for running inferences on the Grounding Dino model.

        Args:
            model_config_path (str): pat to the model config
            model_checkpoint_path (str): path to the weigts
            label_file (str): path to a file containing the labels you want to check in your photos
            csv_file (str): path to a csv_file storing paths to the images you want to run inferences on
            image_column (str): the name of the image column in the csv_file
            logger (str): python logger for logging warnings and info
        """
        self.__model_config_path = model_config_path
        self.__model_checkpoint_path = model_checkpoint_path
        self.__label_file = label_file
        self.__csv_file = csv_file
        self.__image_column = image_column
        self.__logger = logger
        self.__model = None
        self.__prompt = None
        self.__csv_preprocessor = CsvPreprocessor(self.__csv_file, image_column)
        self.__boxes = None
        self.__logits = None
        self.__phrases = None    

    
    def load_model(self):
        """Function that loads the model using the model_checkpoint_path and the pretrained weights
        """
        self.__model = load_model(self.__model_config_path, self.__model_checkpoint_path)

    def load_labels(self):
        """Function that loads the labels from a txt file and rewrites them in a format that is used by the model.
        """
        self.__txt_labels = utils.get_labels(self.__label_file)

        self.__prompt = '. '.join(self.__txt_labels)
        self.__prompt += '.'


    
    def run_inference(self, image_path: str, BOX_THRESHOLD: float, TEXT_THRESHOLD:float):
        """Function that runs a inference on an image.

        Args:
            image_path (str): path to the image
            BOX_THRESHOLD (float): box_threshold meaning what precentage of the confidence is minimum on the inference
            TEXT_THRESHOLD (float): text_threshold meaning what precentage of the confidence need to be at least in the inference
        """

        image_source, image = load_image(image_path=image_path)

        self.__boxes, self.__logits,  self.__phrases = predict(
            model=self.__model,
            image=image,
            caption=self.__prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD)

    
    def run(self, BOX_THRESHOLD:float, TEXT_THRESHOLD: float):  
        """Function thar runs inferences on all the photos.
        """ 
        self.__logger.info('Reading image labels....')
        images = self.__csv_preprocessor()
        self.load_labels()
        self.__df = pd.DataFrame()
        list_phrases = []
        list_boxes = []
        list_images = []
        for image in images:
            self.__logger.info('Running inference on image ' + image)

            self.run_inference(image, BOX_THRESHOLD=BOX_THRESHOLD, TEXT_THRESHOLD=TEXT_THRESHOLD)

            self.__logger.info('Image got boxes ' + str(self.__phrases))

            list_images.append(image)
            list_phrases.append(str(self.__phrases))
            list_boxes.append(str(self.__boxes.cpu().detach().numpy()))

        self.__df.insert(0, "Image_path", list_images, True)
        self.__df.insert(1, "Labels", list_phrases, True)
        self.__df.insert(2, "Boxes", list_boxes, True)

        self.__df.to_csv('b.csv', index=False)

    def __call__(self, BOX_THRESHOLD: float, TEXT_THRESHOLD: float):
        """Make the class callable amd by that we call the whole process of running the inferences.
        Args:
            BOX_THRESHOLD (float): box_threshold meaning what precentage of the confidence is minimum on the inference
            TEXT_THRESHOLD (float): text_threshold meaning what precentage of the confidence need to be at least in the inference
        """
        self.load_model()
        self.run(BOX_THRESHOLD=BOX_THRESHOLD, TEXT_THRESHOLD=TEXT_THRESHOLD)


    