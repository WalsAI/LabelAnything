import clip
from PIL import Image
from utils import utils
import torch
from torch import IntTensor, LongTensor, Tensor
from Preprocessors.CsvPreprocessor import CsvPreprocessor
import shutil
from typing import Tuple, Union
from logging import Logger
import numpy as np


class Inference:
    def __init__(self,
                 folder_path: str,
                 model_type: str,
                 file: str,
                 csv_file: str,
                 image_column: str,
                 device: str,
                 logger: Logger) -> None:
        """
            Class for Inference for categorization with Clip model.
        :param folder_path: path to output directory (place for directories containing images with that specific label)
        :param model_type: type of CLIP model (check CLIP repository)
        :param file: labels file
        :param csv_file: csv file storing image paths
        :param image_column: csv colummn containging image paths
        :param device: name of device to run inference on (currently compatible with cuda and mps)
        :param logger: python logger for logging warnings and info
        """
        self.__model_type = model_type
        self.__file = file
        self.__device = device
        self.__preprocessor = None
        self.__model = None
        self.__logger = logger
        self.__images_label = None
        self.__folder = folder_path
        self.__csv_preprocessor = CsvPreprocessor(csv_file, image_column)

    def get_file(self) -> str:
        """
            Getter for labels file.
        :return: str
        """
        return self.__file

    def get_device(self) -> str:
        """
            Getter for device type.
        :return: str
        """
        return self.__device

    def get_model_type(self) -> str:
        """
            Get clip model type.
        :return: str
        """
        return self.__model_type

    def load_model(self) -> None:
        """
            Load CLIP model.
        :return: CLIP model, preprocessor
        """
        self.__model, self.__preprocessor = clip.load(self.__model_type, device=self.__device)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
            Opening and preprocessing the image from image_path.
        :param image_path: path to the image given
        :return: image as a torch tensor
        """
        image = self.__preprocessor(Image.open(image_path)).unsqueeze(0).to(self.__device)

        return image

    def tokenize_text(self) -> Union[IntTensor, LongTensor]:
        """
            Tokenize text from labels file.
        :return: text tokenized
        """
        text = clip.tokenize(utils.get_labels(self.__file)).to(self.__device)

        return text

    def preprocess_data(self, image_path: str) -> Tuple[Tensor, Union[IntTensor, LongTensor]]:
        """
            Preprocess text and image
        :param image_path: str
        :return: image and text preprocessed.
        """
        image = self.preprocess_image(image_path)
        text = self.tokenize_text()

        return image, text

    def get_probs(self, image: Tensor, text: Tensor) -> np.ndarray:
        """
            Get label probabilities for that specific image.
        :param image: image after preprocessing
        :param text: text after tokenizing
        :return: probabilities for each label
        """
        with torch.no_grad():
            logits_per_image, logits_per_text = self.__model(image, text)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()

        return probs

    def run_inference(self, image_path: str) -> np.ndarray:
        """
            Run inference on a single image
        :param image_path: image path
        :return: probabilities for each label
        """
        image, text = self.preprocess_data(image_path)
        probs = self.get_probs(image, text)

        return probs

    def run(self) -> None:
        """
            Run the inference on all the images from csv file
        :return: None
        """
        images = self.__csv_preprocessor()
        self.__images_label = list()
        translator = utils.translate_labels(self.__file)
        self.__logger.info('Creating label folders...')
        utils.create_label_folders(self.__folder, utils.get_labels(self.__file))

        for image in images:
            self.__logger.info('Running inference on image ' + image)
            probs = self.run_inference(image)
            label = torch.argmax(torch.from_numpy(probs), dim=1)
            self.__images_label.append(label.item())
            self.__logger.info('Image got label ' + translator[label.item()])
            shutil.move(image, self.__folder + '/' + translator[label.item()] + '/' + image.split('/')[-1])

    def __call__(self) -> None:
        """
            Make the class callable
        :return: None
        """
        self.load_model()
        self.run()
