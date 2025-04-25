import os
import sys

from transformers.training_args import TrainingArguments
sys.path.append(os.path.abspath('.'))
from transformers import TrainerCallback, TrainerControl, TrainerState
from llava.model.eval_model.gptscore_model import LLaVA_GPTScoreModel
from llava.model.eval_model.gptscore_model import default_question_template, default_answer_template
from my_datasets.winoground import Winoground
import torch
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

WINOGROUND_ROOT_DIR = '/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets'

# TODO : possibly add an option for the caption score template
def evaluate_winoground(model, tokenizer, data_args, training_args):
    eval_model = LLaVA_GPTScoreModel(
        existing_model=model,
        existing_tokenizer=tokenizer
    )
    
    dataset = Winoground(
        root_dir=WINOGROUND_ROOT_DIR, 
        image_preprocess=eval_model.image_processor.preprocess, 
        return_image_paths=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=training_args.per_device_eval_batch_size, 
        shuffle=False, pin_memory=True)
    
    question_template = default_question_template
    answer_template = default_answer_template
    
    num_samples = len(dataset)
    num_images = len(dataset[0]['images'])
    num_texts = len(dataset[0]['texts'])
    scores = torch.zeros(num_samples, num_images, num_texts).to(eval_model.device)
    
    counter = 0
    for batch in tqdm(loader, total=len(loader), desc="Evaluating Winoground"):
        images = [b['pixel_values'][0] for b in batch['images']]
        texts = batch['texts']
        for i in range(num_images):
            for j in range(num_texts):
                scores[counter:counter+training_args.per_device_eval_batch_size, i, j] = eval_model.forward(
                    images[i], texts[j],
                    question_template=question_template,
                    answer_template=answer_template,
                    neg_answer_template=None)
        counter += training_args.per_device_eval_batch_size

    results, _ = dataset.evaluate_scores(scores)
    return results, scores

class WinogroundEvalCallback(TrainerCallback):
    def __init__(self, eval_steps, data_args, training_args):
        self.eval_steps = eval_steps
        self.data_args = data_args
        self.training_args = training_args

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            model = kwargs.get('model')
            model.eval()
            tokenizer = kwargs.get('tokenizer')
            with torch.no_grad():
                results, scores = evaluate_winoground(model, tokenizer, self.data_args, self.training_args)
            
            # Store results to be logged in the next on_log call
            self.step_results = {f"winoground_{key}": value for key, value in results['all'].items()}
            logger.info(f"Step {state.global_step} Winoground results: {self.step_results}")
            model.train()
        return control
    
    # Evaluate on the first step
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get('model')
        model.eval()
        tokenizer = kwargs.get('tokenizer')
        with torch.no_grad():
            results, scores = evaluate_winoground(model, tokenizer, self.data_args, self.training_args)
        
        # Store initial results to be logged in the next on_log call
        self.initial_results = {f"winoground_{key}": value for key, value in results['all'].items()}
        logger.info(f"Step {state.global_step} Winoground results: {self.initial_results}")

        model.train()
        return control
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if hasattr(self, 'initial_results'):
            logs.update(self.initial_results)
            del self.initial_results
        elif hasattr(self, 'step_results'):
            logs.update(self.step_results)
            del self.step_results
        return control