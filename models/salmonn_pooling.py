import torch
import torch.nn.functional as F
from models.salmonn import SALMONN

class SALMONNPooling(SALMONN):
    def _average_pooling(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                kernel = 1  
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)

                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    speech_embeds_tr = speech_embeds.transpose(1, 2)
                    speech_embeds_overlap = F.avg_pool1d(speech_embeds_tr, kernel_size=kernel, stride=stride)
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, 1, L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(B, -1, C)

                return speech_embeds, kernel
            else:
                raise NotImplementedError
    
    def _qformer_forward(self, speech_embeds):
        with self.maybe_autocast():
            speech_embeds = speech_embeds.reshape(-1, 1, speech_embeds.shape[2])
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)
            
            query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )
            speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

            if self.window_level_Qformer:
                speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
            
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            return speech_embeds, speech_atts
    
    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            B, _, _ = speech_embeds.shape
            speech_embeds, kernel = self._average_pooling(speech_embeds, audio_embeds)
            return self._qformer_forward(speech_embeds)

    def get_speech_embeds(self, spectrogram):
        with self.maybe_autocast():
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
            return speech_embeds
    
    def get_audio_embeds(self, raw_wav, audio_padding_mask):
        with self.maybe_autocast():
            audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            return audio_embeds
    
    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            speech_embeds = self.get_speech_embeds(spectrogram)
            
            if self.beats_path and raw_wav is not None:
                audio_embeds = self.get_audio_embeds(raw_wav, audio_padding_mask)
            else:
                audio_embeds = None
                        
        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)
    
    def forward(self, samples, verbose=False):
        feature = samples.get("feature", None)
        if feature is None:
            return super().forward(samples, verbose=verbose)
        
        speech_embeds, speech_atts = self._qformer_forward(feature)
        
        # wrap speech_embeds with prompts
        if self.prompt_dict:
            prompt = self.prepare_prompt(samples)
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        outputs, correct, total = self.get_output_from_llm(samples["text"], speech_embeds, speech_atts, verbose=verbose)
        loss = outputs.loss

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}