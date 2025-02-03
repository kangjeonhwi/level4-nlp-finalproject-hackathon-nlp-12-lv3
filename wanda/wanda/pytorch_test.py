import torch

# .pt 파일 경로를 지정합니다.
pt_file = "datasets/input_embeds/inputs_embeds_1.pt"
attn_file = "datasets/attention_mask_embeds/attention_mask_1.pt"

device = torch.device("cpu")

# inps 텐서 로드 (.pt 파일)
inps = torch.load(pt_file, map_location=device)
batch_size, seq_length, hidden_size = inps.shape
attention_mask = torch.load(attn_file, map_location=device)
position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
position_embeddings = torch.zeros(batch_size, seq_length, hidden_size, device=device, dtype=inps.dtype)

print("inps shape:", inps.shape)
print("attention_mask shape:", attention_mask.shape)
print("position_ids shape:", position_ids.shape)
print("position_embeddings shape:", position_embeddings.shape)