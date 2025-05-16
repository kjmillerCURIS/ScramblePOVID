import os
import sys
import copy
import uuid
import subprocess
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from PIL import Image

from update_augmentation_policy_type_sensitive import update_augmentation_policy_type_sensitive
from failure_log.data import HNC_CATEGORIES, HNC_TEST
from models import get_model
from clip_finetune.train import main as train_main
from clip_finetune.clipora.config import TrainConfig, parse_yaml_to_config


#load and return data_source, which will have everything needed for doing data synthesis (e.g. original captions)
def load_data(params):
    assert(False)


#Step 1: generate the initial data and return it (and also save it)
#params is some object that has the hyperparameters, probably just a dict
def generate_initial_data(params, data_source):
    assert(False)


#setup "stupid" model and initialize its weights
def setup_model(params):
    if params["model_type"] == "CLIP":
        return get_model(params["model_type"], params["device"])
    elif params["model_type"] == "CLIP_FT":
        return get_model(params["model_type"], params["device"], checkpoint_path=params["checkpoint_path"])
    else:
        raise NotImplementedError

#return initial augmentation_policy - uniform ? 
def initialize_augmentation_policy(params):

    uniform_weight = 1.0 / len(HNC_CATEGORIES)

    augmentation_policy = {
        "mixing_weights": {category: uniform_weight for category in HNC_CATEGORIES}
    }

    return augmentation_policy



#Step 2: use data from current "buffer" to finetune the "stupid" model
#return finetuned model

# Right now this just finetunes on Scramble data
#Some considerations to make the whole process automatic
#1. need to change data path each time to new csv 
#2. need to figure out which model checkpoint to use for loading

def finetune_model(params, data_buffer, model):
    train_script = os.path.abspath("clip_finetune/train.py")
    config_path = os.path.abspath("clip_finetune/train_config.yml")

    config = parse_yaml_to_config(config_path)
    
    train_main(config)

    return None



def get_model_results(params, data_buffer, model):

    def compute_metrics(results):
        correct_counts = defaultdict(int)
        total_counts = defaultdict(int)
        total_correct = 0
        total_samples = 0

        for item in results:
            aug_type = item["aug_type"]
            pos = item["positive_cossim"]
            neg = item["negative_cossim"]

            if pos > neg:
                correct_counts[aug_type] += 1
                total_correct += 1
            total_counts[aug_type] += 1
            total_samples += 1

        metrics = {}
        for aug_type in total_counts:
            metrics[aug_type] = correct_counts[aug_type] / total_counts[aug_type]

        metrics["overall"] = total_correct / total_samples

        return metrics

    
    batch_size = params["eval_batch_size"]
    results = []

    # Set up tokenizer and preprocessor
    tokenizer = model.get_tokenizer()
    preprocess = model.get_image_preprocessor()

    for category in HNC_CATEGORIES:
        eval_data = HNC_TEST(category=category)

        dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

        for (image_paths,), (pos_captions, neg_captions) in tqdm(dataloader, desc="Evaluating"):
            # Preprocess images
            images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
            images = torch.stack(images).to(model.device)

            # Tokenize captions
            pos_tokens = tokenizer(pos_captions).to(model.device)
            neg_tokens = tokenizer(neg_captions).to(model.device)

            # Forward pass
            with torch.no_grad():
                pos_scores = model(images, pos_tokens)
                neg_scores = model(images, neg_tokens)

            # Store results
            for i in range(len(image_paths)):
                results.append({
                    "aug_type": category,                      # use the category for the current dataset
                    "positive_cossim": pos_scores[i, i].item(),
                    "negative_cossim": neg_scores[i, i].item()
                })


    metrics = compute_metrics(results)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    return results



#Step 3: run model on data in data_buffer, use results to update augmentation_policy, which will be returned
def update_augmentation_policy(params, data_buffer, model, augmentation_policy):
    results = get_model_results(params, data_buffer, model)
    if params['type_sensitive']:
        augmentation_policy = update_augmentation_policy_type_sensitive(params, results, augmentation_policy)
    else:
        print('Error: have not yet implemented type-agnostic policy update!')
        assert(False)

    return augmentation_policy


#Step 4: use augmentation policy to augment data, add/replace to existing data_buffer, which will be returned
def augment_data(params, data_source, data_buffer, augmentation_policy):
    assert(False)


#does a run of the full pipeline
#params is some object that has the hyperparameters, probably just a dict
#run_id to be used in any output and intermediate filenames
def run_pipeline(params, run_id):

    # For now remove these steps 
    # data_source = load_data(params)
    # augmentation_policy = initialize_augmentation_policy(params)
    # data_buffer = generate_initial_data(params, data_source)

    model = setup_model(params)
    data_buffer = None
    model = finetune_model(params, data_buffer, model)
    #results = get_model_results(params, data_buffer, model)


    # for t in range(NUM_ITERS):
    #     augmentation_policy = update_augmentation_policy(params, data_buffer, model, augmentation_policy)
    #     data_buffer = augment_data(params, data_source, data_buffer, augmentation_policy)
    #     model = finetune_model(params, data_buffer, model)


if __name__ == "__main__":
    #Need to figure out project structure - right now I am assuming code is run from the project root 
    
    #Model can be "CLIP" or "CLIP_FT"
    params = {
        "model_type": "CLIP_FT",
        "checkpoint_path": "clip_finetune/runs/output_r8_a16_iter1/checkpoint_val_30000",
        "type_sensitive": True,
        "eval_batch_size": 32,
        "device": torch.device("cuda"),
        # add more parameters as needed
    }

    run_id = str(uuid.uuid4())  # generate a unique ID for this run
    run_pipeline(params, run_id)