import hydra
import omegaconf
import sys
import os

sys.path.append(os.getcwd())
sys.path.append('..')

from Preprocessors.Preprocess import Preprocess
from Preprocessors.FolderTransformer import FolderTransformer
#from CLIP.Inference import Inference
from grounding_dino.inference import DinoInference
from utils.utils import configure_logger


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg) -> None:
    """
        Main function for categorization
    :param cfg:
    :return:
    """
#    logger = configure_logger(cfg.CLIP.logger_name, cfg.CLIP.file_handler)

    # preprocessor = Preprocess(cfg.CLIP.path_to_folder, cfg.CLIP.csv_column, cfg.CLIP.csv_name, logger)
    # reprocessor()


    # inferencer = Inference(cfg.CLIP.output_folder, cfg.CLIP.model_type, cfg.CLIP.labels, cfg.CLIP.csv_name, cfg.CLIP.csv_column, cfg.CLIP.device, logger)
    # inferencer()

    preprocess_dino = FolderTransformer(cfg.DINO.path_to_folder, cfg.DINO.csv_column, cfg.DINO.csv_name)
    preprocess_dino()

    inferencer_dino = DinoInference(cfg.DINO.model_config_path, cfg.DINO.model_checkpoint_path, cfg.DINO.label_file, cfg.DINO.csv_name, cfg.DINO.csv_column, cfg.DINO.logger_name)
    inferencer_dino(float(cfg.DINO.box_threshold), float(cfg.DINO.text_threshold))
     
    

    

if __name__ == '__main__':
    main()
