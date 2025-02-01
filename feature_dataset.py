from dataset import SALMONNDataset

class SALMONNFeatureDataset(SALMONNDataset):
    def __init__(self, prefix, ann_path, whisper_path):
        super().__init__(prefix, ann_path, whisper_path)
        self.ann_path = ann_path