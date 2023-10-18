import clip
from PIL import Image
import utils
import torch
from Preprocessors.CsvPreprocessor import CsvPreprocessor

class Inference:
    def __init__(self, model_type, file, csv_file, image_column, device):
        self.__model_type = model_type
        self.__file = file
        self.__device = device
        self.__preprocessor = None
        self.__model = None
        self.__csv_preprocessor = CsvPreprocessor(csv_file, image_column)

    def get_file(self):
        return self.__file

    def get_device(self):
        return self.__device

    def get_model_type(self):
        return self.__model_type

    def preprocess_image(self, image_path):
        pass

    def load_model(self):
        self.__model, self.__preprocessor = clip.load(self.__model_type, device=self.__device)

    def preprocess_image(self, image_path):
        image = self.__preprocessor(Image.open(image_path)).unsqueeze(0).to(self.__device)

        return image

    def tokenize_text(self):
        text = clip.tokenize(utils.get_labels(self.__file)).to(self.__device)

        return text

    def preprocess_data(self, image_path):
        image = self.preprocess_image(image_path)
        text = self.tokenize_text()

        return image, text

    def get_features(self, image, text):
        with torch.no_grad():
            image_features = self.__model.encode_image(image)
            text_features = self.__model.encode_text(text)

        return image_features, text_features

    def get_probs(self, image, text):
        with torch.no_grad():
            logits_per_image, logits_per_text = self.__model(image, text)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()

        return probs

    def run_inference(self, image_path):
        image, text = self.preprocess_data(image_path)
        probs = self.get_probs(image, text)

        return probs

    def run(self):
        images = self.__csv_preprocessor()
        images_label = list()

        for image in images:
            probs = self.run_inference(image)
            label = torch.argmax(probs, dim=0)
            images_label.append(label)

        return images_label

    def __call__(self):

        return self.run()
