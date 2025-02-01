import os
import time
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
        
        prefix = loader.prefix
        
        start_time = time.time()
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
            
            tensor_path = os.path.join(prefix, f"{sample['id'][0]}.pt")
            torch.save(pooled_embeds, tensor_path)
            
            time_for_extraction += time.time() - start_time