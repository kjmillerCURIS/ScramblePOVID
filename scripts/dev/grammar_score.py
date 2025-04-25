import os
import sys
sys.path.append(os.path.abspath('.'))

import transformers
import torch


def load_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained('textattack/distilbert-base-uncased-CoLA')
    model = transformers.DistilBertForSequenceClassification.from_pretrained('textattack/distilbert-base-uncased-CoLA')
    model.to(device='cpu')
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model
    }
    
@torch.no_grad()
def get_score(caption, tokenizer, model):
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
    score = output['logits'].softmax(dim=1)[:, 1]
            
    return score[0].item()

