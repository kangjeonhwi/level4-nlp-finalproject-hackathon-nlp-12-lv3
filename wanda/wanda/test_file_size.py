import os
import torch

inps_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/input_embeds"
attn_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/attention_mask_embeds"

out_inps_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/output_inps_1"
out_attn_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/output_atts_1"

os.makedirs(out_inps_directory, exist_ok=True)
os.makedirs(out_attn_directory, exist_ok=True)

def convert_first_batch_only(src_dir, dst_dir):
    files = sorted(os.listdir(src_dir))
    for fname in files:
        if not fname.endswith(".pt"):
            continue
        fpath = os.path.join(src_dir, fname)
        data = torch.load(fpath, map_location="cpu")  # shape: [8, 88, 2048] or [8, 88]
        
        # 첫 번째 항만 추출: data[0:1] → shape [1, 88, 2048] or [1, 88]
        new_data = data[0:1, ...]
        
        # 저장할 경로
        outpath = os.path.join(dst_dir, fname)
        torch.save(new_data, outpath)
        print(f"{fpath} -> {outpath}, shape={new_data.shape}")

# 실제 변환 실행
convert_first_batch_only(inps_directory, out_inps_directory)
convert_first_batch_only(attn_directory, out_attn_directory)
