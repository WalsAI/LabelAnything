from segment_anything import sam_model_registry, SamPredictor

class SAM:
    def __init__(self, sam_weights, sam_type, device):
        self.__sam_weights = sam_weights
        self.__sam_type = sam_type
        self.__device = device
    
    def __call__(self):
        sam = sam_model_registry[self.__sam_type](checkpoint=self.__sam_weights)
        sam.to(self.__device)

        predictor = SamPredictor(sam)

        return predictor