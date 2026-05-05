import json

path = 'scripts/train/train_multimodal_text_guided_cross_attn.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if 'from src.models.backbones.text.clip_text_encoder import CLIPTextEncoder' not in source and 'from src.models.losses.wrapper_loss import InfoNCESupConLoss' in source:
            source = source.replace('from src.models.losses.wrapper_loss import InfoNCESupConLoss\n', 'from src.models.losses.wrapper_loss import InfoNCESupConLoss\nfrom src.models.backbones.text.clip_text_encoder import CLIPTextEncoder\n')
            cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
            
        if 'def create_training_components' in source:
            old_str = 'def create_training_components(config: Dict):\n    device = resolve_device(config["training"]["device"])\n'
            new_str = 'def create_training_components(config: Dict):\n    device = resolve_device(config["training"]["device"])\n    text_encoder = CLIPTextEncoder(\n        model_name=config["model"]["clip_model_name"],\n        pretrained=config["model"]["clip_pretrained"],\n        device=device,\n        normalize=True\n    )\n'
            source = source.replace(old_str, new_str)
            source = source.replace('        clip_model_name=config["model"]["clip_model_name"],\n        clip_pretrained=config["model"]["clip_pretrained"],\n', '')
            source = source.replace('return model, criterion, optimizer, scheduler, device', 'return model, text_encoder, criterion, optimizer, scheduler, device')
            cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
            
        if 'def train_epoch' in source:
            source = source.replace('def validate_epoch(model, dataloader, criterion, device="cpu"):', 'def validate_epoch(model, text_encoder, dataloader, criterion, device="cpu"):')
            source = source.replace('outputs = model(images, texts)', 'text_feat = text_encoder(texts)\n        outputs = model(images, text_feat)')
            source = source.replace('def train_epoch(model, dataloader, criterion, optimizer, device="cpu"):', 'def train_epoch(model, text_encoder, dataloader, criterion, optimizer, device="cpu"):')
            cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
            
        if 'model, criterion, optimizer, scheduler, device = create_training_components(config)' in source:
            source = source.replace('model, criterion, optimizer, scheduler, device = create_training_components(config)', 'model, text_encoder, criterion, optimizer, scheduler, device = create_training_components(config)')
            cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
            
        if 'def train_text_guided_contrastive' in source:
            source = source.replace('model, criterion, optimizer, scheduler, device = create_training_components(config)', 'model, text_encoder, criterion, optimizer, scheduler, device = create_training_components(config)')
            source = source.replace('train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)', 'train_metrics = train_epoch(model, text_encoder, train_loader, criterion, optimizer, device)')
            source = source.replace('val_metrics = validate_epoch(model, val_loader, criterion, device)', 'val_metrics = validate_epoch(model, text_encoder, val_loader, criterion, device)')
            cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print('Patched successfully!')
