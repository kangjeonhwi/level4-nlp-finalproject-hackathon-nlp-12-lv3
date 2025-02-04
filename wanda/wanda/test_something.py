import os

inps_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/output_inps_1"
attn_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/output_atts_1"

inps_files = sorted(os.listdir(inps_directory))
attn_files = sorted(os.listdir(attn_directory))

print("inps file count =", len(inps_files))
print("attn file count =", len(attn_files))