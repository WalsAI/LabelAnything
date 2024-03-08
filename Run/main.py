import hydra
from omegaconf import DictConfig
import sys
import os
from hydra import initialize, compose
from omegaconf import OmegaConf


sys.path.append(os.getcwd())
sys.path.append('..')

from Preprocessors.Preprocess import Preprocess
from Preprocessors.FolderTransformer import FolderTransformer
#from CLIP.Inference import Inference
from grounding_dino.inference import DinoInference
from utils.utils import configure_logger


def main() -> None:
    """
        Main function for running diferent types of labeling (categorization, object detection and segmentation)
    :param cfg: DictConfig
    :return: None
    """

    with initialize(version_base=None, config_path="../conf"):
        cfg_model = compose(config_name="config")
        cfg = compose(config_name="config_hydra")

    model = cfg.ConfHydra
    if model == "Classification":
        model = cfg_model.CLIP

        logger = configure_logger(model.logger_name, model.file_handler)

        preprocessor = Preprocess(model.path_to_folder, model.csv_column, model.csv_name, logger)
        preprocessor()

        inferencer = Inference(model.output_folder, model.model_type, model.labels, model.csv_name, model.csv_column, model.device, logger)
        inferencer()
    elif model == "Detection":
        model = cfg_model.DINO
        grounding_dino_logger = configure_logger(model.logger_name, model.file_handler)

        preprocess_dino = FolderTransformer(model.path_to_folder, model.csv_column, model.csv_name)
        preprocess_dino()

        inferencer_dino = DinoInference(model.model_config_path, model.model_checkpoint_path, model.label_file, model.csv_name, model.csv_column, grounding_dino_logger)
        inferencer_dino(float(model.box_threshold), float(model.text_threshold))
     
    

    

if __name__ == '__main__':
    main()
