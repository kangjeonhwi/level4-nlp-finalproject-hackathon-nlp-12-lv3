import os
import time
import json
import torch
from tqdm import tqdm
from utils import prepare_sample
from runner import Runner


class Extractor(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def extract(self, mode="train"):
        loader = getattr(self, f"{mode}_loader", None)
        assert loader is not None, f"Invalid mode: {mode}"
        
        self.model.eval()
        
        time_for_speech_processing = 0.0
        time_for_audio_processing = 0.0
        time_for_pooling = 0.0
        time_for_extraction = 0.0
        
        # print("Attention!")
        if mode == "train":
            prefix = loader._dataloader.dataset.prefix
            ann_path = loader._dataloader.dataset.ann_path
        else:
            prefix = loader.dataset.prefix
            ann_path = loader.dataset.ann_path
        
        start_time = time.time()
        new_data_json = []
        print(len(loader))
        count = 0
        samples = 0
        for sample in tqdm(loader):
            sample = prepare_sample(sample, cuda_enabled=self.cuda_enabled)
            
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
            pooled_embeds, _ = self.model._average_pooling(speech_embeds, audio_embeds)
            time_for_pooling += time.time() - start_time
            
            start_time = time.time()
            print(pooled_embeds.shape)
            for i in range(pooled_embeds.shape[0]):
                tensor_path = f"{sample['id'][i]}.pt"
                # print(pooled_embeds[i].shape)
                torch.save(pooled_embeds[i].cpu(), prefix + "/" + tensor_path)
                new_data_json.append({
                    "path": sample['id'][i],
                    "feature": tensor_path,
                    "task": sample["task"][i],
                    "Q": sample["Q"][i],
                    "text": sample["text"][i],
                })
                time_for_extraction += time.time() - start_time
            #     print(pooled_embeds[i].element_size() * pooled_embeds[i].nelement())
            # print("end")
            
            count += 1
            samples += pooled_embeds.shape[0]
            if count == len(loader):
                break
        
        ann_path_splited = ann_path.split(".")
        ann_path_splited[-2] += "_extracted"
        new_ann_path = ".".join(ann_path_splited)
        with open(new_ann_path, "w") as f:
            json.dump({"annotation": new_data_json}, f, indent=4)
        
        ann_path_splited[-2] += "-meta"
        meta_path = ".".join(ann_path_splited)
        time_for_speech_processing /= samples
        time_for_audio_processing /= samples
        time_for_pooling /= samples
        time_for_extraction /= samples
        
        with open(meta_path, "w") as f:
            json.dump({
                "time_for_speech_processing": time_for_speech_processing,
                "time_for_audio_processing": time_for_audio_processing,
                "time_for_pooling": time_for_pooling,
                "time_for_extraction": time_for_extraction,
                "samples" : samples
            }, f, indent=4)
        
        print(f"Time for speech processing (Avg.): {time_for_speech_processing:.2f}")
        print(f"Time for audio processing (Avg.): {time_for_audio_processing:.2f}")
        print(f"Time for pooling (Avg.): {time_for_pooling:.2f}")
        print(f"Time for extraction (Avg.): {time_for_extraction:.2f}")
        