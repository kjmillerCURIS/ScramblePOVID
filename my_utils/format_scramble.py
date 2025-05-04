import json
import csv

# Paths and filenames
INPUT_JSON = "/projectnb/ivc-ml/ac25/Compositional_Reasoning/ScramblePOVID/data/preference_data/coco_train_syn_cot_adv_ref_preference.json"
OUTPUT_CSV = "/projectnb/ivc-ml/ac25/Compositional_Reasoning/ScramblePOVID/data/preference_data/coco_train_syn_cot_adv_ref_preference.csv"
IMAGE_PREFIX = "/projectnb/ivc-ml/array/data/COCO/images/train2017/"

def extract_caption(convo_list):
    for message in convo_list:
        if message["from"] == "gpt":
            return message["value"]
    return ""  # fallback

def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    with open(OUTPUT_CSV, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "positive_caption", "negative_caption"])

        for item in data:
            image_path = IMAGE_PREFIX + item["image"]
            pos_caption = extract_caption(item.get("conversations", []))
            neg_caption = extract_caption(item.get("rejected_conversations", []))
            writer.writerow([image_path, pos_caption, neg_caption])

if __name__ == "__main__":
    main()
