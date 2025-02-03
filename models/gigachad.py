# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM, ATModel
from .modeling_whisper import WhisperModel
from .salmonn import SALMONN
from .utils import StoppingCriteriaSub


class GIGACHAD(SALMONN):
    def __init__(
        self,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        
        n_layer=32, 
        rep_dim=1280, 
        tltr_mode='tl_down_tr_512_1_8',

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        use_at_model=True,
        freeze_at_model = False,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        token=None,
        only_preprocessor=None,
    ):
        super().__init__()

        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource
        
        self.use_at_model = use_at_model
        self.freeze_at_model = freeze_at_model

        self.ln_audio = nn.LayerNorm(rep_dim)

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False, token=token)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        if use_at_model:
            self.at_model = ATModel(
                n_layer=n_layer, 
                rep_dim=rep_dim, 
                mode=tltr_mode
            )
        
        if not only_preprocessor:
            logging.info('Loading LLaMA Model')
            if self.low_resource:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map={"": device_8bit},
                    token=token,
                )
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16,
                    token=token,
                )

            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            logging.info('Loading LLaMA Done')

            if self.lora:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=lora_rank, 
                    lora_alpha=lora_alpha, 
                    lora_dropout=lora_dropout,
                )
                self.llama_model = get_peft_model(self.llama_model, self.peft_config)
                self.llama_model.print_trainable_parameters()
                logging.info('LoRA Training')

        assert whisper_path
        logging.info('Loading Whisper Model')
        whisper_config = AutoConfig.from_pretrained(whisper_path)
        whisper_config.output_hidden_states = True  # 중간 레이어 출력 활성화
        self.speech_encoder = WhisperModel.from_pretrained(
            whisper_path, 
            config=whisper_config
        ).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        if freeze_whisper:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logging.info("freeze Whisper")
        

        if self.use_speech_Qformer:
            self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                num_query_token=num_speech_query_token,
                speech_width=self.speech_encoder.config.d_model + self.at_model.rep_dim 
            )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            logging.info('Loading speech LLAMA proj')
            if only_preprocessor:
                config = AutoConfig.from_pretrained(llama_path, token=token)
                lm_hidden_size = config.hidden_size
            else:
                lm_hidden_size = self.llama_model.config.hidden_size
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, lm_hidden_size
            )
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            # Whisper 중간 레이어 추출
            encoder_outputs = self.speech_encoder(
                spectrogram, 
                return_dict=True
            )
            speech_embeds = encoder_outputs.last_hidden_state  # 최종 레이어
            all_hidden = encoder_outputs.hidden_states  # 모든 레이어 [num_layers+1, B, T, D]

            # ATModel 입력 형식 맞춤 (B, num_layers, T, D)
            hidden_states = torch.stack(all_hidden[1:], dim=1)  # 임베딩 제외
            B, num_layers, T, D = hidden_states.shape
            audio_embeds = self.at_model(hidden_states)  # [B, T, D]
            audio_embeds = self.ln_audio(audio_embeds)

        return self._encode_auditory_feature(speech_embeds, audio_embeds)

    @classmethod
    def from_config(cls, config):
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)
        
        use_at_model = config.get("use_at_model", True)
        freeze_at_model = config.get("freeze_at_model", False)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)
        
        n_layer = config.get("n_layer", 32)
        rep_dim = config.get("rep_dim", 1280)  
        tltr_mode = config.get("tltr_mode", "tl_down_tr_512_1_8")

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        token = config.get("token", None)
        only_preprocessor = config.get("only_preprocessor", None)

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            n_layer=n_layer,
            rep_dim=rep_dim,
            tltr_mode=tltr_mode,
            use_at_model=use_at_model,
            freeze_at_model=freeze_at_model,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            token=token,
            only_preprocessor=only_preprocessor,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load GIGACHAD ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)

        return model
