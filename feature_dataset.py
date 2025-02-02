import torch
from torch.nn.utils.rnn import pad_sequence
from dataset import SALMONNDataset

class SALMONNFeatureDataset(SALMONNDataset):
    def __init__(self, prefix, ann_path, whisper_path):
        super().__init__(prefix, ann_path, whisper_path)
        self.ann_path = ann_path
        
    def collater(self, samples):
        have_feature = [s.get("feature", None) is not None for s in samples]
        
        if not (all(have_feature) or not any(have_feature)):
            raise ValueError("Either all samples should have feature or none of them")
        
        if not all(have_feature):
            return super().collater(samples)
        
        features = [s["feature"] for s in samples]
        features_len = torch.tensor([len(s["feature"]) for s in samples])
        features = pad_sequence(features, batch_first=True, padding_value=0)
        padding_mask = torch.arange(features.size(1)).unsqueeze(0) >= features_len.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]
        
        return {
            "feature": features,
            "padding_mask": padding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }
        
    def __getitem__(self, index):
        ann = self.annotation[index]
        feature_path = self.prefix + '/' + ann["feature"]
        try:
            feature = torch.load(feature_path)
            return {
                "feature": feature,
                "text": ann["text"],
                "task": ann.get("task", "asr"),
                "Q": ann.get("Q", ""),
                "id": ann["path"],
            }
        except:
            return super().__getitem__(index)