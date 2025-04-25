import os
import sys
sys.path.append(os.path.abspath('.'))

import transformers
import torch

# @torch.autocast(device_type='cuda', dtype=torch.float16)
def load_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera', device_map='auto')
    model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera', device_map='auto')
    model.D = model.shared.embedding_dim
    linear = torch.nn.Linear(model.D, 1, dtype=model.dtype, device=model.device)
    linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
    linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
    model.eval()
    t = model.shared.weight[32097, 0].item() # temperature for calibration
    return {
        "tokenizer": tokenizer,
        "model": model,
        "linear": linear,
        "t": t
    }

@torch.no_grad()
# @torch.autocast(device_type='cuda', dtype=torch.float16)
def get_score(caption, tokenizer, model, linear, t):
    toks = tokenizer.batch_encode_plus(
        [caption], 
        return_tensors='pt', 
        padding='longest', 
        truncation='longest_first', 
        max_length=128
    )
    input_ids = toks.input_ids.to(model.device)
    attention_mask = toks.attention_mask.to(model.device)
        
    output = model(input_ids, attention_mask=attention_mask)
    last_indices = attention_mask.sum(dim=1, keepdim=True) - 1
    last_indices = last_indices.unsqueeze(-1).expand(-1, -1, model.D)
    last_hidden_state = output.last_hidden_state.gather(dim=1, index=last_indices).squeeze(1)
    logit = linear(last_hidden_state).squeeze(-1)
    logit_calibrated = logit / t
    score_calibrated = logit_calibrated.sigmoid()
    
    return score_calibrated[0].item()
