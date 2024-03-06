import pandas as pd

class SAMINference:
    def __init__(self, sam_model, csv_path, csv_column, bboxes_preprocessor):
        self.__csv_path = csv_path
        self.__csv_column = csv_column
        self.__bboxes_preprocessor = bboxes_preprocessor
        self.__sam_model = sam_model
        self.__images_csv = None
        self.__columns_dict = None

    def read_image_csv(self):
        self.__images_csv = pd.read_csv(self.__csv_path)

        return self.__images_csv[self.__csv_column]

    def preprocess_boxes(self):
        boxes_dict, self.__columns_dict = self.__bboxes_preprocessor()

        return boxes_dict

    def concatenate_masks(self):
        pass

    def output_mask(self, image):
        boxes_dict = self.preprocess_boxes()
        masks, _, _ = self.__sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_dict[image][None, :],
            multimask_output=False
        )

        return masks

    def run_inference_on_image(self, image):
        self.__sam_model.set_image(image)
        masks = self.output_mask(image)

        return masks
        
        