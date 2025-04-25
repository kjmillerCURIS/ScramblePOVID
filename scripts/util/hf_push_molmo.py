import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import torch
from huggingface_hub import login

from transformers import AutoProcessor, AutoModelForCausalLM
from my_datasets.eqben_mini import image_loader

def main(args):
    # Model initialization
    model_path = os.path.expanduser(args.model_path)
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True, 
    )
    
    if not args.repo_id:
        print("Error: --repo_id must be specified.") 
        sys.exit(1)
        
    print(f"\nAttempting to push model and processor to Hugging Face Hub repository: {args.repo_id}")
    print("Make sure you are logged in. Run 'huggingface-cli login' or set the HUGGING_FACE_HUB_TOKEN environment variable.")
    
    try:
        print(f"Pushing model to {args.repo_id}...")
        model.push_to_hub(args.repo_id)
        
        print(f"Pushing processor to {args.repo_id}...")
        processor.push_to_hub(args.repo_id)
        
        print("\nSuccessfully pushed model and processor to Hugging Face Hub!")
        print(f"Access your repository at: https://huggingface.co/{args.repo_id}")
        sys.exit(0)

    except Exception as e:
        print(f"\nAn error occurred during push_to_hub: {e}")
        print("Please ensure you are logged in (huggingface-cli login), the repo_id is correct,")
        print("and the repository exists or can be created by your user.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a Molmo model and push it to Hugging Face Hub.")
    parser.add_argument("--model-path", type=str, default='../trl-new/checkpoint/molmo_train_coco_train_syn_cot_adv_ref_lora2/', help="Path to the local Molmo model directory")
    parser.add_argument("--repo_id", type=str, default='samarthm44/SCRAMBLe-Molmo-7B-D-0924', 
                        help="The repository ID (e.g., 'your-username/your-model-name') to push to on Hugging Face Hub.")
    args = parser.parse_args()
    
    main(args)