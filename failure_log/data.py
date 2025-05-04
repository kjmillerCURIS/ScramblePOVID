import os
import json
from torch.utils.data import Dataset

HNC_CATEGORIES = [
    "attribute", "attribute_relation", "relation", "relation_attribute", "object_count", 
    "object_compare_count", "verify_object_attribute", "verify_object_relation", 
    "and_logic_attribute", "and_logic_relation", "xor_logic_attribute", "xor_logic_relation"
]

current_dir = os.path.dirname(__file__)
CAPTION_PATH = os.path.join(current_dir, "test_set_v1-0.json")
IMAGE_PATH = os.path.join(current_dir, "test_images")

class HNC_TEST(Dataset):
    def __init__(self, category, image_path=IMAGE_PATH, caption_path=CAPTION_PATH):
        assert category in HNC_CATEGORIES, f"Unknown category: {category}"
        
        self.dataset_joint = {}  # grouped by image
        self.dataset = []        # flat list of (image, pos, neg) triplets
        self.image_path = image_path
        self.category = category
        
        with open(caption_path, "r") as f:
            caption_data = json.load(f)

        for image_key, captions in caption_data.items():
            pos_list = []
            neg_list = []

            for caption_info in captions.values():
                if caption_info["type"] != category:
                    continue
                if caption_info["label"] == "1":
                    pos_list.append(caption_info["caption"])
                elif caption_info["label"] == "0":
                    neg_list.append(caption_info["caption"])
            
            if pos_list or neg_list:
                if len(pos_list) != len(neg_list):
                    print(f"Warning: Unmatched positive and/or negative captions for image {image_key} and category {category}. +1 example discarded")
                    continue

                image_file = os.path.join(image_path, f"{image_key}.jpg")
                self.dataset_joint[image_file] = {
                    "pos_captions": pos_list,
                    "neg_captions": neg_list
                }

                for pos_caption, neg_caption in zip(pos_list, neg_list):
                    self.dataset.append((image_file, pos_caption, neg_caption))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_file, caption_pos, caption_neg = self.dataset[idx]
        return (image_file,), (caption_pos, caption_neg)


# for category in HNC_CATEGORIES:
#         dataset = HNC_TEST(category=category)
#         print(f"Category: {category}, Num: {len(dataset)}")


dataset = HNC_TEST(category="xor_logic_attribute")
print(dataset[1])

