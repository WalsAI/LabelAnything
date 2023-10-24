import logging
import numpy as np
import cv2
import os


def open_image(image_path: str) -> np.ndarray:
    '''
        Open an image given its path using opencv
    :param image_path: str
    :return: np.ndarray
    '''

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# Temporary function for getting the labels - later will be converted to '.txt', '.yml', '.yaml' - will be deleted later
def get_labels(file):
    labels = None
    if file.endswith(".txt"):
        file = open(file, 'r')
        labels = file.readline().split(", ")

    return labels


def translate_labels(file):
    labels = get_labels(file)

    return {i: labels[i] for i in range(len(labels))}


def configure_logger(logger_name, file_handler):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(file_handler, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

def create_label_folders(folder_path, labels):
    for label in labels:
        path = folder_path + '/' + label
        if not os.path.exists(path):
            os.mkdir(path)
