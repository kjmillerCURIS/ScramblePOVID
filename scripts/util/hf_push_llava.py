import argparse
import torch
import os
from PIL import Image
import readline  # Add this import for command history

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def setup_history(history_file):
    # Set up command history
    histfile = os.path.expanduser(history_file)
    try:
        readline.read_history_file(histfile)
        # Default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(1000)
    except FileNotFoundError:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(histfile), exist_ok=True)
    
    # Save history on exit
    import atexit
    atexit.register(readline.write_history_file, histfile)

def main(args):
    # Determine history file path
    history_file = args.history_file
    # Set up command history
    setup_history(history_file)
    
    # Model initialization
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name or get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    
    # Define your desired repo ID on the Hub
    repo_id = "samarthm44/SCRAMBLe-llava-v1.5-13b" # Replace with your details
    
    try:
        print(f"Attempting to push model to {repo_id}...")
        model.push_to_hub(repo_id)
        
        print(f"Attempting to push tokenizer to {repo_id}...")
        tokenizer.push_to_hub(repo_id)
        
        if image_processor: # Check if image_processor exists
            print(f"Attempting to push image processor to {repo_id}...")
            image_processor.push_to_hub(repo_id)
            
        print("\nSuccessfully pushed components to Hugging Face Hub!")
        print(f"Access your model at: https://huggingface.co/{repo_id}")

    except AttributeError as e:
        print(f"\nError: push_to_hub method not found. {e}")
        print("It seems the loaded objects might not be standard Hugging Face objects,")
        print("or the method is unavailable for this specific model type.")
        print("Consider using the 'save_pretrained' and 'huggingface_hub.upload_folder' method instead.")
        
    except Exception as e:
        print(f"\nAn error occurred during push_to_hub: {e}")
        print("Please ensure you are logged in (huggingface-cli login) and the repo_id is correct.")
    
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()
    
    main(args)
