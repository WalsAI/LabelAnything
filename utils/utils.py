import logging
import numpy as np
import cv2
import os
from typing import List, Dict
from logging import Logger


def open_image(image_path: str) -> np.ndarray:
    """
        Open an image given its path using opencv
    :param image_path: str
    :return: np.ndarray
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def get_labels(file: str) -> List[str]:
    """
        Get the labels from file (currently supporting only txt extension)
    :param file: path to the file containing the labels
    :return: list of labels
    """
    labels = None
    if file.endswith(".txt"):
        file = open(file, 'r')
        labels = file.readline().split(", ")

    return labels


def translate_labels(file: str) -> Dict[int, str]:
    """
        Translate labels from integer to string
    :param file: path to the file containing the labels
    :return: Dict containing the translation
    """
    labels = get_labels(file)

    return {i: labels[i] for i in range(len(labels))}


def configure_logger(logger_name: str, file_handler: str) -> Logger:
    """
        Configure the logger
    :param logger_name: name of the logger
    :param file_handler: path to the file that will be used for logging
    :return: Logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(file_handler, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def create_label_folders(folder_path: str, labels: List[str]) -> None:
    """
        Create folder for each label in the file containing them
    :param folder_path: path to the directory where the subdirectories will be created
    :param labels: list of labels
    :return: None
    """
    for label in labels:
        path = folder_path + '/' + label
        if not os.path.exists(path):
            os.mkdir(path)
