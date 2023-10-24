import hydra
import omegaconf
import sys
import os

sys.path.append(os.getcwd())
sys.path.append('..')

from Preprocessors.Preprocess import Preprocess
from CLIP.Inference import Inference
from utils.utils import configure_logger


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg) -> None:
    """
        Main function for categorization
    :param cfg:
    :return:
    """
    logger = configure_logger(cfg.CLIP.logger_name, cfg.CLIP.file_handler)

    preprocessor = Preprocess(cfg.CLIP.path_to_folder, cfg.CLIP.csv_column, cfg.CLIP.csv_name, logger)
    preprocessor()

    inferencer = Inference(cfg.CLIP.output_folder, cfg.CLIP.model_type, cfg.CLIP.labels, cfg.CLIP.csv_name, cfg.CLIP.csv_column, cfg.CLIP.device, logger)
    inferencer()


if __name__ == '__main__':
    main()
