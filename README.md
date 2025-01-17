# SALMONN: Speech Audio Language Music Open Neural Network

<div align=center><img src="resource/salmon.png" height="256px" width="256px"/></div>

<h1 align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com/?lines=Hello,+There!+👋;Welcome+to+SALMONN;&center=true&size=30">
  </a>
</h1>

🚀🚀 Welcome to the repo of **SALMONN**!

SALMONN is a large language model (LLM) enabling **speech, audio events, and music inputs**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. Instead of speech-only input or audio-event-only input, SALMONN can perceive and understand all kinds of audio inputs and therefore obtain emerging capabilities such as multilingual speech recognition and translation and audio-speech co-reasoning. This can be regarded as giving the LLM "ears" and cognitive hearing abilities, which makes SALMONN a step towards hearing-enabled artificial general intelligence.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://bytedance.github.io/SALMONN/'><img src='https://img.shields.io/badge/SALMONN_13B-Demo-blue'></a>
<a href='https://huggingface.co/spaces/tsinghua-ee/SALMONN-7B-gradio'><img src='https://img.shields.io/badge/SALMONN_7B-Demo-orange'></a>
<a href='https://openreview.net/pdf?id=14rn7HpKVk'><img src='https://img.shields.io/badge/SALMONN_paper-PDF-green'></a>
<a href='https://openreview.net/pdf?id=nYsh5GFIqX'><img src='https://img.shields.io/badge/video_SALMONN_paper-PDF-green'></a>
<a href='https://huggingface.co/tsinghua-ee/SALMONN'><img src='https://img.shields.io/badge/huggingface-checkpoint-yellow'></a> 
</div>

## 🌟 Structure

The model architecture of SALMONN is shown below. A window-level Q-Former is used as the connection module to fuse the outputs from a Whisper speech encoder and a BEATs audio encoder as augmented audio tokens, which are aligned with the LLM input space. The LoRA adaptor aligns the augmented LLM input space with its output space. The text prompt is used to instruct SALMONN to answer open-ended questions about the general audio inputs and the answers are in the LLM text responses. 

<div align=center><img src="resource/structure.png" height="100%" width="75%"/></div>

## ⚡️ Demos

Compared with traditional speech and audio processing tasks such as speech recognition and audio caption, SALMONN leverages the general knowledge and cognitive abilities of the LLM to achieve a cognitively oriented audio perception, which dramatically improves the versatility of the model and the richness of the task. In addition, SALMONN is able to follow textual commands and even spoken commands with a relatively high degree of accuracy. Since SALMONN only uses training data based on textual commands, listening to spoken commands is also a cross-modal emergent ability.

Here are some examples of SALMONN.

| Audio                                                  | Response                                     |
| ------------------------------------------------------ | -------------------------------------------- |
| [gunshots.wav](./resource/audio_demo/gunshots.wav)     | ![sac](resource/response_demo/sac.png)       |
| [duck.wav](./resource/audio_demo/duck.wav)             | ![story](resource/response_demo/story.png)   |
| [music.wav](./resource/audio_demo/music.wav)           | ![mc](resource/response_demo/mc.png)         |

## Datasets
* Download raw audio files from [here](https://huggingface.co/datasets/lifelongeeek/salmonn_train_stage1_stage2)
  * Put downloaded directory path into `data_prefix` of config
  * contains 1.4TB of audio
    ```
    168G  WavCaps
    165G  audiocaps
    110G  GigaSpeech
    58G   LibriSpeech
    3.7G  MusicNet
    2.0G  Clotho
    ```
* Download annotation files from [here](https://huggingface.co/datasets/lifelongeeek/salmonn_dataset_annotation)
  * place the jsons under `data` directory.
  * NOTE: Only train split will be released to public.

## 🌈 How to train a model

For SALMONN-13B v1, you need to use the following dependencies:
1. Our environment: The python version is 3.9.17, and other required packages can be installed with the following command: ```pip install -r requirements.txt```.
2. Download [whisper large v2](https://huggingface.co/openai/whisper-large-v2/tree/main) to ```whisper_path```.
3. Download [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) to `beats_path`.
4. Download [vicuna 13B v1.1](https://huggingface.co/lmsys/vicuna-13b-v1.1/tree/main) to ```llama_path```.
5. Running with ```python3 train.py --cfg-path configs/config.yaml```
6. You may try `--dryrun` for loading dataset and dummy small model.

## 🌈 How to inference in CLI

1. Same as **How to train a model: 1-4**.
2. Download [salmonn v1](https://huggingface.co/tsinghua-ee/SALMONN/blob/main/salmonn_v1.pth) to ```ckpt```.
3. Running with ```python3 cli_inference.py --cfg-path configs/decode_config.yaml``` Now you can input ```wav_path``` and ```prompt```. Enjoy yourself !

## 🌈 How to launch a web demo

1. Same as **How to train a model: 1-4**.
2. Download [salmonn v1](https://huggingface.co/tsinghua-ee/SALMONN/blob/main/salmonn_v1.pth) to ```ckpt```.
3. Running with ```python3 web_demo.py --cfg-path configs/decode_config.yaml```

## 👀 Team

**Team Tsinghua**: Wenyi Yu, Changli Tang, Guangzhi Sun, Chao Zhang

**Team ByteDance**: Xianzhao Chen, Wei Li, Tian Tan, Lu Lu, Zejun Ma

## ✨ Citation
If you find SALMONN / video-SALMONN useful, please cite the paper:
```
@inproceedings{
  tang2024salmonn,
  title={{SALMONN}: Towards Generic Hearing Abilities for Large Language Models},
  author={Changli Tang and Wenyi Yu and Guangzhi Sun and Xianzhao Chen and Tian Tan and Wei Li and Lu Lu and Zejun MA and Chao Zhang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=14rn7HpKVk}
}
```
---
# Audiolm Evaluator
Audio Language Model Evaluator

## Install dependencies
```bash
git clone --recursive https://github.com/nota-github/audiolm-evaluator
pip install -r audiolm-trainer/requirements.txt
pip install -r requirements.txt
aac-metrics-download
```

## Evaluate
`salmonn_eval_config.yaml` 에서 데이터셋 경로, 모델 경로 등을 적절히 수정한 후 아래 스크립트를 실행합니다.
```python
python evaluate_salmonn.py --mode {submission_asr, submission_aac, valid_asr, valid_aac}
```
- submission mode는 제출용인 csv를 만들기 위한 모드입니다.
- valid mode는 자체적인 평가를 진행하고자 할 때 사용하며 text 라벨이 있는 json 파일이 필요합니다.
- 두 모드는 서로 다른 디렉토리에 csv 파일이 저장됩니다.

```
{
  "annotation": [
    {
      "testset_id": "any_id_for_test",
      "path": "/path/to/audio_file",
      "task": {asr or audiocaption_v2},
      "text": "Ground truth for sample" # valid 시 필요
    },
    ...
```

## Validate submission file
```python
python submission_validator.py /path/to/submission.csv
```

위 스크립트는 파일의 형식만 확인하며, 샘플의 개수는 validation하지 않습니다.