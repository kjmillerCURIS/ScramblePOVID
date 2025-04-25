from typing import List, Union
import torch
import copy

from transformers import AutoProcessor, AutoModelForCausalLM
# from .mm_utils import expand2square, tokenizer_image_token
from llava.constants import SYSTEM_MSG, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.utils import disable_torch_init

default_question_template = "Does the image show '{}'? Please answer yes or no."
default_answer_template = "Yes"

class GPTScoreModel():
    def __init__(self,
                 model_path=None,
                 model_base=None,
                 model_name=None,
                 device='cuda',
                 existing_model=None,
                 existing_tokenizer=None):
        
        if existing_model is not None:
            self.model = existing_model
            self.processor = existing_tokenizer  # Note: Changed to processor
            self.device = existing_model.device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
            self.processor.chat_template = self.processor.tokenizer.chat_template
            # self.model.to(device=device, dtype=torch.bfloat16)
            self.device = device
        
        self.conversational_style = 'chat'

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def get_score(
        self,
        images: Union[List[str], torch.Tensor],
        texts: List[str],
        question_template: str=default_question_template,
        answer_template: str=default_answer_template,
    ) -> torch.Tensor:
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Questions and answers are formatted in preprocess
        questions = [[{'role': 'user', 'content': question}, {'role': 'assistant', 'content': ''}] for question in questions]
        questions = [self.processor.apply_chat_template(q) for q in questions]
        # questions = [' ' + q for q in questions] # for always_start_with_space
        
        # questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        prompts = [qs + ans for qs, ans in zip(questions, answers)]
        # tokens = self.processor.tokenizer(prompts, padding=True, return_tensors='pt')['input_ids']

        # TODO : figure out how to get assistant locations for actually training
        # Process inputs using the LLaVA processor
        inputs = []
        for img, prompt in zip(images, prompts):
            curr_inp = self.processor.process(
                images=[img], 
                text=prompt,
                return_tensors="pt", 
                # padding=True,
                message_format=None,
            )
            inputs.append(curr_inp)
        max_len = max([len(i['input_ids']) for i in inputs])
        
        for i in inputs:
            i['input_ids'] = torch.cat([i['input_ids'], self.processor.tokenizer.pad_token_id * torch.ones(max_len - len(i['input_ids']), dtype=torch.long)], dim=0)
        
        final_inputs = {}
        for k in inputs[0]:
            final_inputs[k] = torch.stack([i[k] for i in inputs]).to(device=self.device)
        
        inputs = final_inputs
        
        # Prepare labels
        labels = copy.deepcopy(inputs['input_ids'])
        
        for img, label, qs in zip(images, labels, questions):
            tokenized_len = len(self.processor.process(images=[img], text=qs, return_tensors='pt', message_format=None)['input_ids'])
            if qs[-1] == " ":
                tokenized_len -= 1
            label[:tokenized_len] = IGNORE_INDEX
            label[label == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

        outputs = self.model(
            **inputs,
            labels=labels,
            return_dict=True
        )

        logits = outputs.logits
        
        # Calculate log probabilities
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_log_prob = torch.zeros(shift_logits.shape[0])
        for k in range(lm_log_prob.shape[0]):
            lm_log_prob[k] = (-loss_fct(shift_logits[k], shift_labels[k]))
        return lm_log_prob


    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self,
                images: Union[List[str], torch.Tensor],
                texts: List[str],
                question_template: str=default_question_template,
                answer_template: str=default_answer_template,
                neg_answer_template: str=None) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        if neg_answer_template is not None:
            pos_score = self.get_score(images, texts, question_template, answer_template)
            neg_score = self.get_score(images, texts, question_template, neg_answer_template)
            return pos_score - neg_score
        else:
            return self.get_score(images, texts, question_template, answer_template)