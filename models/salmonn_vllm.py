import os
import torch
from models.salmonn import SALMONN
from vllm import LLM, SamplingParams
from peft import PeftModel

class SALMONN_VLLM(SALMONN):
    def __init__(self, *args, vllm_lora_path=None, vllm_llm_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_lora_path = vllm_lora_path
        self.vllm_llm_path = vllm_llm_path
        self.llama_vocab_size = self.llama_model.config.vocab_size
    
    def _is_lora_saved(self):
        if not os.path.exists(self.vllm_lora_path):
            return False
        return True

    def _save_lora(self):
        assert self.lora and self.llama_model is PeftModel
        self.llama_model.save_pretrained(self.vllm_lora_path)
    
    def _is_vllm_llm_saved(self):
        if not os.path.exists(self.vllm_llm_path):
            return False
        return True
    
    def _save_vllm_llm(self):
        assert self.lora and self.llama_model is PeftModel
        self.llama_model = self.llama_model.merge_and_unload()
        self.llama_model.save_pretrained(self.vllm_llm_path)
        
    def _load_vllm(self):
        if not self._is_vllm_llm_saved():
            self._save_vllm_llm()
        
        return LLM(
            model = self.vllm_llm_path,
        )
    
    def _encode_prompt(self, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(self.llama_model.device)
                
                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(self.llama_model.device)
            else:
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(self.llama_model.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(self.llama_model.device)
            return p_before_tokens.input_ids, p_after_tokens.input_ids
        else:
            return torch.tensor([[]]), torch.tensor([[]])
    
    def generate(self, samples, generate_cfg, prompt=None):
        # Q-Former에 대한 결과값을 얻는다.
        
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        if prompt is None and self.test_prompt_dict:
            if self.multi_prompt:
                prompt = [self.test_prompt_dict[task] for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = self.prompt_dict[samples["task"][0]]

        speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=True)

        batch_size, query_per_sample = speech_embeds.shape[:2]
        
        speech_embeds_viewed = speech_embeds.view((-1,) + speech_embeds.shapes[-2:])
        num_new_tokens = speech_embeds_viewed.size(0)
        
        current_embeds = self.llama_model.get_input_embeddings()
        original_embeds = current_embeds.weight.data[:self.llama_vocab_size]

        extended_embeds = torch.cat([original_embeds, speech_embeds_viewed], dim=0)
        self.llama_model.get_input_embeddings().weight.data = extended_embeds
        # resizing_token_embeddings() 함수를 안쓰는 이유: output_embeds도 같이 바뀐다.
        
        prompt_tokens = self._encode_prompt(prompt, multi_prompt=True)
        p_before_tokens, p_after_tokens = prompt_tokens
        query_tokens = torch.arange(self.llama_vocab_size, self.llama_vocab_size + num_new_tokens, device=self.llama_model.device)
        query_tokens = query_tokens.view(batch_size, query_per_sample)
        bos = torch.ones((batch_size, 1), dtype=p_before_tokens.dtype, device=p_before_tokens.device) * self.llama_tokenizer.bos_token_id
        
        vllm_input_tokens = torch.cat([bos, p_before_tokens, query_tokens, p_after_tokens], dim=1)

        sampling_params = SamplingParams(
            max_tokens=generate_cfg.get("max_new_tokens", 200),
            n=1,
            beam_width=generate_cfg.get("num_beams", 4), 
            use_beam_search=not generate_cfg.get("do_sample", False), 
            min_tokens=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
        )
        
        llm = self._load_vllm()
        outputs = llm.generate(vllm_input_tokens, sampling_params)
        generated_text = outputs[0].outputs[0].text 
        
        return generated_text
    
    @classmethod
    def parse_config(cls, config):
        config_dict = super().parse_config(config)
        config_dict["vllm_lora_path"] = config.get("vllm_lora_path", "")
        config_dict["vllm_llm_path"] = config.get("vllm_llm_path", "")
        return config_dict