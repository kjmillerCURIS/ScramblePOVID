from typing import List, Union
import torch
import copy

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
# from .mm_utils import expand2square, tokenizer_image_token
from llava.constants import SYSTEM_MSG, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

from llava.utils import disable_torch_init

default_question_template = "Does the image show '{}'? Please answer yes or no."
default_answer_template = "Yes"

# TODO : look into expand2square vs resize transform

# NOTE : These formattings are according to conv_vicuna_v1 from conversation.py
def format_question(question, conversation_style='chat'):
    if conversation_style == 'plain': # for 1st stage model
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 'chat': # for 2nd stage model
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    else:
        raise NotImplementedError()
    return question

def format_answer(answer, conversation_style='chat'):
    if conversation_style == 'plain': # for 1st stage model
        answer = answer + "\n"
    elif conversation_style == 'chat': # for 2nd stage model
        answer = answer + "</s>"
    else:
        raise NotImplementedError()
    return answer

class LLaVA_GPTScoreModel():
    """A wrapper for the LLaVA-1.5 models"""
    def __init__(self,
                 model_path=None,
                 model_base=None,
                 model_name=None,
                 device='cuda',
                 existing_model=None,
                 existing_tokenizer=None):
        disable_torch_init()
        
        self.conversational_style = 'chat'
        if existing_model is not None:
            self.model = existing_model
            self.tokenizer = existing_tokenizer
            self.image_processor = existing_model.get_vision_tower().image_processor
            self.device = existing_model.device
        else:      
            # NOTE : model_name should have 'lora' for loading lora weights and it should have 'llava' to be a llava model type
            self.model_path = model_path
            self.model_base = model_base
            self.model_name = model_name if model_name is not None else get_model_name_from_path(model_path)
            self.device = device
            
            # self.model should typically be of type LlavaLlamaForCausalLM
            
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=model_base,
                model_name=model_name,
                device=device,
            )
            # self.model.get_vision_tower().to(dtype=self.model.dtype, device=self.model.device)

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def get_score(
        self,
        images: Union[List[str], torch.Tensor],
        texts: List[str],
        question_template: str=default_question_template,
        answer_template: str=default_answer_template,
    ) -> torch.Tensor:
        self.model : LlavaLlamaForCausalLM # just so that pylance knows the type of model    
        
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Is the image showing 'a photo of a dog'? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Formatting for LLaVA-1.5 desired input including system message and image tokens
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        # NOTE : Sometimes performance improves when format_answer is removed so that end_token generation is not considered
        # answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]
        
        # if isinstance(images, (list, tuple)): # if list of image paths
        #     images = self.load_images(images)
        
        prompts = [qs + ans for qs, ans in zip(questions, answers)]
        
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in prompts]
        labels = copy.deepcopy(input_ids)
        
        # assert all(len(qs) == len(questions[0]) for qs in questions), 'currently prefix only works for fixed question'
        for label, qs in zip(labels, questions):
            tokenized_len = len(tokenizer_image_token(qs, self.tokenizer))
            if qs[-1] == " ":
                tokenized_len -= 1 # because white space
            label[:tokenized_len] = IGNORE_INDEX
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
            
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
        input_ids, _, attention_mask, past_key_values, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None, # position_ids
            attention_mask,
            None, # past_key_values
            labels,
            images
        )
        
        # assert input_ids is None, "input_ids should be None for LLaVA-1.5"
        # assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
        # model_input_kwargs = {
        #     'input_ids': input_ids, # None for LLaVA-1.5
        #     'attention_mask': attention_mask,
        #     'past_key_values': past_key_values,
        #     'inputs_embeds': inputs_embeds,
        #     'use_cache': None,
        #     'output_attentions': None,
        #     'output_hidden_states': None,
        #     'return_dict': False,
        # }
        
        # outputs = self.model.model(
        #     **model_input_kwargs
        # )

        # hidden_states = outputs[0]
        # logits = self.model.lm_head(hidden_states)
        
        outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True
        )

        logits = outputs.logits

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        shift_labels = shift_labels.to(shift_logits.device)
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