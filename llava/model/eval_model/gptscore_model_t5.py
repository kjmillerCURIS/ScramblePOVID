from typing import List, Union
import torch
import copy

import transformers

from llava.model.builder import load_pretrained_model_t5
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, t5_tokenizer_image_token, expand2square
from llava.model import CLIPT5ForConditionalGeneration
# from .mm_utils import expand2square, tokenizer_image_token
from llava.constants import SYSTEM_MSG, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, CONTEXT_LEN, HF_CACHE_DIR
# NOTE : context length is 2048

from llava.utils import disable_torch_init

from transformers import T5TokenizerFast

from dataclasses import dataclass, field
from typing import Optional

from torchvision.datasets.folder import default_loader

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="clip-flant5-xxl")
    version: Optional[str] = field(default="t5_v1")
    # freeze_backbone: bool = field(default=False)
    # tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_patch_merge_type: Optional[str] = field(default='flat')

# default_question_template = "Does the image show '{}'? Please answer yes or no."
default_question_template = "Does this figure show '{}'? Please answer yes or no."
default_answer_template = "Yes"

# TODO : look into expand2square vs resize transform

# # NOTE : These formattings are according to conv_vicuna_v1 from conversation.py
# def format_question(question, conversation_style='chat'):
#     if conversation_style == 'plain': # for 1st stage model
#         question = DEFAULT_IMAGE_TOKEN + question
#     elif conversation_style == 'chat': # for 2nd stage model
#         question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
#     else:
#         raise NotImplementedError()
#     return question

# def format_answer(answer, conversation_style='chat'):
#     if conversation_style == 'plain': # for 1st stage model
#         answer = answer + "\n"
#     elif conversation_style == 'chat': # for 2nd stage model
#         answer = answer + "</s>"
#     else:
#         raise NotImplementedError()
#     return answer

# NOTE : the question format used in the end is the same as LlaVA-1.5
def format_question(question, conversation_style='plain'):
    if conversation_style == 't5_plain': # for 1st stage t5 model
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 't5_chat': # for 2nd stage t5 model
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system': # for 2nd stage t5 model
        question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system_no_user': # for 2nd stage t5 model
        question = "" + DEFAULT_IMAGE_TOKEN + "\n" + question + " : "
    # elif conversation_style == 't5_chat_ood_system': # for 2nd stage t5 model
    #     question = SYSTEM_MSG + " HUMAN: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " GPT: "
    else:
        raise NotImplementedError()
    return question

def format_answer(answer, conversation_style='plain'):
    return answer

class T5_GPTScoreModel():
    """A wrapper for the T5 models"""
    def __init__(self,
                 model_path=None,
                 model_base=None,
                 model_name=None,
                 device='cuda',
                 existing_model=None,
                 existing_tokenizer=None):
        # disable_torch_init()
        
        self.conversational_style = 't5_chat'
        if existing_model is not None:
            self.model = existing_model
            self.tokenizer = existing_tokenizer
            # self.image_processor = existing_model.get_vision_tower().image_processor
            self.device = existing_model.device
        else:      
            # NOTE : model_name should have 'lora' for loading lora weights and it should have 'llava' to be a llava model type
            self.model_path = model_path
            self.model_base = model_base
            self.model_name = model_name if model_name is not None else get_model_name_from_path(model_path)
            self.device = device
            self.cache_dir = HF_CACHE_DIR
            
            # model_args = ModelArguments()
            # model_max_length = CONTEXT_LEN
            # padding_side = None
            # mmprojector_repo = None
            # mmprojector_name = None
            
            # # default is 'pad'
            # # stage-1 models use 'square'
            # self.image_aspect_ratio = 'pad'
            # self.conversational_style = 't5_chat'
            
            # self.image_loader = default_loader
            
            # self.context_len = CONTEXT_LEN
            
            # self.tokenizer, self.model, self.image_processor = load_pretrained_model_t5(
            #     CLIPT5ForConditionalGeneration,
            #     model_args,
            #     model_path=self.model_path,
            #     tokenizer_path='google/flan-t5-xxl',
            #     model_max_length=model_max_length,
            #     padding_side=padding_side,
            #     image_aspect_ratio=self.image_aspect_ratio,
            #     mmprojector_repo=mmprojector_repo,
            #     mmprojector_name=mmprojector_name,
            #     device=self.device,
            #     cache_dir=self.cache_dir
            # )
            
            config = transformers.AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = CLIPT5ForConditionalGeneration.from_pretrained(
                model_path,
                config=config,
                device_map=device,
                # dtype=torch.float16
            )
            self.model.to(dtype=torch.float16)
            model_args = ModelArguments()
            model_args.model_name_or_path = self.model_path
            self.model.get_model().initialize_vision_modules(model_args=model_args)
            self.model.get_vision_tower().to(dtype=self.model.dtype, device=self.model.device)
            # vision_tower = self.model.get_vision_tower()
            # if not vision_tower.is_loaded:
            #     vision_tower.load_model()
            # vision_tower.to(dtype=self.model.dtype, device=self.device)
            self.image_processor = self.model.get_vision_tower().image_processor
            self.tokenizer = T5TokenizerFast.from_pretrained(
                self.model_path,
                model_max_length=512,
                truncation_side='right',
                padding_side="right",
            )
            
            self.image_processor = self.model.get_vision_tower().image_processor

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        if self.image_aspect_ratio == 'pad':
            image = [expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean)) for image in image]
        image = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def get_score(
        self,
        images: Union[List[str], torch.Tensor],
        texts: List[str],
        question_template: str=default_question_template,
        answer_template: str=default_answer_template,
    ) -> torch.Tensor:
        self.model : CLIPT5ForConditionalGeneration # just so that pylance knows the type of model    
        
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Is the image showing 'a photo of a dog'? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Formatting for LLaVA-1.5 desired input including system message and image tokens
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        # NOTE : Sometimes performance improves when format_answer is removed so that end_token generation is not considered
        answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]
        
        # if isinstance(images, (list, tuple)): # if list of image paths
        #     images = self.load_images(images)
        
        # prompts = [qs + ans for qs, ans in zip(questions, answers)]
        
        input_ids = [t5_tokenizer_image_token(qs, self.tokenizer, return_tensors='pt') for qs in questions]
        labels = [t5_tokenizer_image_token(ans, self.tokenizer, return_tensors='pt') for ans in answers]
        
        # input_ids = [t5_tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in prompts]
        # labels = copy.deepcopy(input_ids)
        
        # assert all(len(qs) == len(questions[0]) for qs in questions), 'currently prefix only works for fixed question'
        # for label, qs in zip(labels, questions):
        #     tokenized_len = len(tokenizer_image_token(qs, self.tokenizer))
        #     if qs[-1] == " ":
        #         tokenized_len -= 1 # because white space
        #     label[:tokenized_len] = IGNORE_INDEX
    
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
        decoder_attention_mask = labels.ne(IGNORE_INDEX)
        
        input_ids, attention_mask, decoder_attention_mask, labels = input_ids.to(self.device), \
            attention_mask.to(self.device), decoder_attention_mask.to(self.device), labels.to(self.device)
        model_input_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'images': images,
            'past_key_values': None,
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': True,
        }
        
        outputs = self.model(
            **model_input_kwargs
        )

        logits = outputs.logits
        lm_log_prob = torch.zeros(logits.shape[0])
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        for k in range(lm_log_prob.shape[0]):
            lm_log_prob[k] = (-loss_fct(logits[k], labels[k]))
        return lm_log_prob
        
        # """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        # """
        # assert len(images) == len(texts), "Number of images and texts must match"
        # # Turn "a photo of a dog" into
        # # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # # A: "Yes"
        # questions = [question_template.format(text) for text in texts]
        # answers = [answer_template.format(text) for text in texts]
        
        # # Formatting for CLIP-FlanT5 desired input including system message and image tokens
        # questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        # answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]

        # # images = self.load_images(images)
        
        # input_ids = [t5_tokenizer_image_token(qs, self.tokenizer, return_tensors='pt') for qs in questions]
        # labels = [t5_tokenizer_image_token(ans, self.tokenizer, return_tensors='pt') for ans in answers]

        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX)
        # input_ids = input_ids[:, :self.tokenizer.model_max_length]
        # labels = labels[:, :self.tokenizer.model_max_length]

        # attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # decoder_attention_mask = labels.ne(IGNORE_INDEX)
        
        # input_ids, attention_mask, decoder_attention_mask, labels = input_ids.to(self.device), \
        #     attention_mask.to(self.device), decoder_attention_mask.to(self.device), labels.to(self.device)
        # model_input_kwargs = {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'decoder_attention_mask': decoder_attention_mask,
        #     'labels': labels,
        #     'images': images,
        #     'past_key_values': None,
        #     'inputs_embeds': None,
        #     'use_cache': None,
        #     'output_attentions': None,
        #     'output_hidden_states': None,
        #     'return_dict': True,
        # }
        
        # outputs = self.model(
        #     **model_input_kwargs
        # )

        # logits = outputs.logits
        # lm_prob = torch.zeros(logits.shape[0])
        # loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # for k in range(lm_prob.shape[0]):
        #     lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp() # exp to cancel the log and get raw prob between 0 and 1
        # return lm_prob


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