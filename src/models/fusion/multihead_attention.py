import torch
import torch.nn as nn

model = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.2, batch_first=True)

print(f"{'Tên thành phần':<25} | {'Loại':<15} | {'Kích thước / Tham số (Input -> Output)'}")
print("-" * 80)

for name, param in model.named_parameters():
    # in_proj là trọng số gộp của Q, K, V
    if 'in_proj_weight' in name:
        print(f"{name:<25} | {'Parameter':<15} | [512] -> [1536] (Gộp Q,K,V)")
    elif 'in_proj_bias' in name:
        print(f"{name:<25} | {'Parameter':<15} | Bias kích thước [1536]")
        
    # out_proj là lớp chiếu cuối cùng
    elif 'out_proj.weight' in name:
        print(f"{name:<25} | {'nn.Linear':<15} | [512] -> [512]")
    elif 'out_proj.bias' in name:
        print(f"{name:<25} | {'Parameter':<15} | Bias kích thước [512]")

print("-" * 80)
print(f"Tổng số tham số: {sum(p.numel() for p in model.parameters()):,}")