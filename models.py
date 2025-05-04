import torch
from clip_finetune.clipora.lora.inject import inject_linear_attention
from clip_finetune.clipora.lora.attention import InjectedMultiHeadAttention
from open_clip import create_model_from_pretrained, get_tokenizer, get_model_config, create_model_and_transforms
from torch.nn import CosineSimilarity
from itertools import product
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel, LoraConfig, get_peft_model


class CLIP_FT():

    def __init__(self, device, checkpoint_path):
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-B-32",
            "openai",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            image_mean=None,
            image_std=None,
        )
        
        model_config = get_model_config("ViT-B-32")

        model = inject_linear_attention(
            model=model,
            encoders={"transformer"},
            embed_dim=model_config["embed_dim"],
            num_heads=model_config["text_cfg"]["heads"],
        )
    
        model = inject_linear_attention(
            model=model,
            encoders={"visual.transformer"},
            embed_dim=model_config["vision_cfg"]["width"],
            num_heads=12,
        )
    
        # Load LoRA-wrapped model + weights
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model.to(device)
        model.eval()
        
        self.model = model
        self.preprocess = preprocess_val
        self.tokenizer = get_tokenizer("ViT-B-32")
        self.device = device

    def get_image_preprocessor(self):
        return self.preprocess

    def get_tokenizer(self):
        return self.tokenizer

    def __call__(self, images, captions):
        # images: [B, 3, H, W]
        # captions: tokenized, [B, T]
        image_embd, text_embd, _ = self.model(images, captions)
        similarity = torch.nn.functional.cosine_similarity(image_embd.unsqueeze(1), text_embd.unsqueeze(0), dim=-1)
        return similarity  # shape [B, B]



class CLIP_BASE():

    def __init__(self, device):
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-B-32",       
            "openai",    #can also change this here to load from open clip library 
            precision="amp",  
            device=device,    
            jit=False,      
            force_quick_gelu=False,
            image_mean=None,
            image_std=None,
        )
        
        model.to(device)
        model.eval()
        
        self.model = model
        self.preprocess = preprocess_val
        self.tokenizer = get_tokenizer("ViT-B-32")
        self.device = device

    def get_image_preprocessor(self):
        return self.preprocess

    def get_tokenizer(self):
        return self.tokenizer

    def __call__(self, images, captions):
        # images: [B, 3, H, W]
        # captions: tokenized, [B, T]
        image_embd, text_embd, _ = self.model(images, captions)
        similarity = torch.nn.functional.cosine_similarity(image_embd.unsqueeze(1), text_embd.unsqueeze(0), dim=-1)
        return similarity  # shape [B, B]


MODELS = {
    "CLIP": CLIP_BASE,
    "CLIP_FT": CLIP_FT
}

def get_model(name, device, **kwargs):
    assert name in MODELS, f"Unknown model: {name}"
    return MODELS[name](device, **kwargs)
