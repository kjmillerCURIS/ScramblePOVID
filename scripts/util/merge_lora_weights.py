import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


# Example usage
# python scripts/util/merge_lora_weights.py --model-path checkpoint/coco_syn/train_coco_train_syn_cot_adv_ref_high_lr/ --model-base liuhaotian/llava-v1.5-13b --model-name llava_lora_coco_syn --save-model-path checkpoint/merged/train_coco_train_syn_cot_adv_ref_high_lr

def merge_lora(args):
    # model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, args.model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True) # NOTE : model_name needs to have 'lora' to correctly identify model class
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
