import os
import sys


#load and return data_source, which will have everything needed for doing data synthesis (e.g. original captions)
def load_data(params):
    assert(False)


#Step 1: generate the initial data and return it (and also save it)
#params is some object that has the hyperparameters, probably just a dict
def generate_initial_data(params, data_source):
    assert(False)


#setup "stupid" model and initialize its weights
def setup_model(params):
    assert(False)


#return initial augmentation_policy
def initialize_augmentation_policy(params):
    assert(False)


#Step 2: use data from current "buffer" to finetune the "stupid" model
#return finetuned model
def finetune_model(params, data_buffer, model):
    assert(False)


#Step 3: run model on data in data_buffer, use results to update augmentation_policy, which will be returned
def update_augmentation_policy(params, data_buffer, model, augmentation_policy):
    assert(False)


#Step 4: use augmentation policy to augment data, add/replace to existing data_buffer, which will be returned
def augment_data(params, data_source, data_buffer, augmentation_policy):
    assert(False)


#does a run of the full pipeline
#params is some object that has the hyperparameters, probably just a dict
#run_id to be used in any output and intermediate filenames
def run_pipeline(params, run_id):
    data_source = load_data(params)
    model = setup_model(params)
    augmentation_policy = initialize_augmentation_policy(params)
    data_buffer = generate_initial_data(params, data_source)
    model = finetune_model(params, data_buffer, model)
    for t in range(NUM_ITERS):
        augmentation_policy = update_augmentation_policy(params, data_buffer, model, augmentation_policy)
        data_buffer = augment_data(params, data_source, data_buffer, augmentation_policy)
        model = finetune_model(params, data_buffer, model)
