import os
import time
import json
import torch
from runner import Runner


class Extractor(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # use single batch
        self.train_loader._dataloader.batch_size = 1
        self.valid_loader.batch_size = 1
        self.test_loader.batch_size = 1
        
    def extract(self, mode="train"):
        loader = getattr(self, f"{mode}_loader", None)
        assert loader is not None, f"Invalid mode: {mode}"
        
        self.model.eval()
        
        time_for_speech_processing = 0.0
        time_for_audio_processing = 0.0
        time_for_pooling = 0.0
        time_for_extraction = 0.0
        
        if mode == "train":
            prefix = loader._dataloader.dataset.prefix
            ann_path = loader._dataloader.dataset.ann_path
        else:
            prefix = loader.dataset.prefix
            ann_path = loader.dataset.ann_path
        
        start_time = time.time()
        new_data_json = []
        for sample in loader:
            
            spectrogram = sample["spectrogram"]
            raw_wav = sample.get("raw_wav", None)
            audio_padding_mask = sample.get("padding_mask", None)
            
            start_time = time.time()
            speech_embeds = self.model.get_speech_embeds(spectrogram)
            time_for_speech_processing += time.time() - start_time
            
            start_time = time.time()
            audio_embeds = self.model.get_audio_embeds(raw_wav, audio_padding_mask)
            time_for_audio_processing += time.time() - start_time
                        
            start_time = time.time()
            pooled_embeds = self.model._average_pooling(speech_embeds, audio_embeds)
            time_for_pooling += time.time() - start_time
            
            start_time = time.time()
            tensor_path = f"{sample['id'][0]}.pt"
            torch.save(pooled_embeds, os.path.join(prefix, tensor_path))
            new_data_json.append({
                "path": sample['id'][0],
                "feature": tensor_path,
                "task": sample["task"][0],
                "Q": sample["Q"][0],
                "text": sample["text"][0],
            })
            time_for_extraction += time.time() - start_time
        
        ann_path_splited = ann_path.split(".")
        ann_path_splited[-2] += "_extracted"
        new_ann_path = ".".join(ann_path_splited)
        with open(os.path.join(prefix, new_ann_path), "w") as f:
            json.dump({"annotation": new_data_json}, f, indent=4)
        
        ann_path_splited[-2] += "-meta"
        meta_path = ".".join(ann_path_splited)
        time_for_speech_processing /= len(loader)
        time_for_audio_processing /= len(loader)
        time_for_pooling /= len(loader)
        time_for_extraction /= len(loader)
        
        with open(os.path.join(prefix, meta_path), "w") as f:
            json.dump({
                "time_for_speech_processing": time_for_speech_processing,
                "time_for_audio_processing": time_for_audio_processing,
                "time_for_pooling": time_for_pooling,
                "time_for_extraction": time_for_extraction,
                "samples" : len(loader)
            }, f, indent=4)
        
        print(f"Time for speech processing (Avg.): {time_for_speech_processing:.2f}")
        print(f"Time for audio processing (Avg.): {time_for_audio_processing:.2f}")
        print(f"Time for pooling (Avg.): {time_for_pooling:.2f}")
        print(f"Time for extraction (Avg.): {time_for_extraction:.2f}")
        