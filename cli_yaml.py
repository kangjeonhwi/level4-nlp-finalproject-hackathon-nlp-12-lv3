# cli_yaml.py

train_template = """\
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

model:
  # paths
  llama_path: "{llama_path}"
  whisper_path: "{whisper_path}"
  beats_path: "{beats_path}"

  token: "{token}" # Use hf token to access gated repositories 

  ckpt: "{ckpt}" # if not "", load model from ckpt for training or evaluation

  freeze_whisper: {freeze_whisper}
  freeze_beats: {freeze_beats}

  # window-level Q-Former
  use_speech_Qformer: {use_speech_Qformer}
  freeze_speech_QFormer: {freeze_speech_QFormer}
  window_level_Qformer: {window_level_Qformer}
  num_speech_query_token: {num_speech_query_token}
  second_per_window: {second_per_window}
  second_stride: {second_stride}

  speech_llama_proj_model: "{speech_llama_proj_model}"
  freeze_speech_llama_proj: {freeze_speech_llama_proj}

  # LoRA
  lora: {lora}
  lora_rank: {lora_rank}
  lora_alpha: {lora_alpha}
  lora_dropout: {lora_dropout}

  multi_prompt: {multi_prompt}
  prompt_template: "{prompt_template}"
  prompt_path: "{prompt_path}"
  test_prompt_path: "{test_prompt_path}"
  max_txt_len: {max_txt_len}
  end_sym: "{end_sym}"

datasets:
  prefix: "{datasets_prefix}"
  train_ann_path: "{train_ann_path}"
  valid_ann_path: "{valid_ann_path}"
  test_ann_path: "{test_ann_path}"
  whisper_path: "{whisper_path}"

run:
  # log & settings
  seed: {seed}
  output_dir: "{output_dir}"
  evaluate: {evaluate} # if True, only evaluate model on test data
  exp_name: "{exp_name}"

  log_freq: {log_freq}
  epoch_based: {epoch_based}
  iters_per_epoch: {iters_per_epoch}
  accum_grad_iters: {accum_grad_iters}
  batch_size_train: {batch_size_train}
  batch_size_eval: {batch_size_eval}
  num_workers: {num_workers}

  device: "{device}"
  use_distributed: {use_distributed}
  amp: {amp}
  world_size: {world_size}
  dist_url: "{dist_url}"

  # optimizer & scheduler
  optims:
    max_epoch: {max_epoch}
    warmup_steps: {warmup_steps}
    warmup_start_lr: {warmup_start_lr}
    init_lr: {init_lr}
    min_lr: {min_lr}
    weight_decay: {weight_decay}
    beta2: {beta2}
"""

eval_template = """\
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

model:
  llama_path: "{llama_path}"
  whisper_path: "{whisper_path}"
  beats_path: "{beats_path}"

  token: "{token}" # Use hf token to access gated repositories
  only_preprocessor: {only_preprocessor}

  ckpt: "{ckpt}" # model used for decoding

  freeze_whisper: {freeze_whisper}
  freeze_beats: {freeze_beats}

  # window-level Q-Former
  use_speech_Qformer: {use_speech_Qformer}
  freeze_speech_QFormer: {freeze_speech_QFormer}
  window_level_Qformer: {window_level_Qformer}
  num_speech_query_token: {num_speech_query_token}
  second_per_window: {second_per_window}
  second_stride: {second_stride}

  speech_llama_proj_model: "{speech_llama_proj_model}"
  freeze_speech_llama_proj: {freeze_speech_llama_proj}

  # LoRA
  lora: {lora}
  lora_rank: {lora_rank}
  lora_alpha: {lora_alpha}
  lora_dropout: {lora_dropout}

  multi_prompt: {multi_prompt}
  prompt_template: "{prompt_template}"
  prompt_path: "{prompt_path}"
  test_prompt_path: "{test_prompt_path}"
  max_txt_len: {max_txt_len}
  end_sym: "{end_sym}"

generate:
  max_new_tokens: {max_new_tokens}
  num_beams: {num_beams}
  do_sample: {do_sample}
  min_length: {min_length}
  temperature: {temperature}
  top_p: {top_p}
  repetition_penalty: {repetition_penalty}
  length_penalty: {length_penalty}
  end_sym: "{end_sym}"

datasets:
  prefix: "{datasets_prefix}"
  test_ann_path_asr: "{test_ann_path_asr}"
  test_ann_path_aac: "{test_ann_path_aac}"
  valid_ann_path: "{valid_ann_path}"
  whisper_path: "{whisper_path}"

run:
  batch_size_eval: {batch_size_eval}
  num_workers: {num_workers}
  device: "{device}"
"""
