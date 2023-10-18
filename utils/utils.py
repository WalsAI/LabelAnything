import numpy as np
import cv2

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
    if file.endswith(".txt"):
        file = open(file, 'r')
        labels = file.readline().split(" ")

    return labels