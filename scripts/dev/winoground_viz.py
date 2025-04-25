import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import torch
from my_datasets.winoground import get_winoground_scores, Winoground
from pathlib import Path
import base64
from PIL import Image
import io
from tqdm import tqdm

def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        img.thumbnail((500, 500))  # Resize image to 500x500 max
        if img.mode in ('RGBA', 'LA'):
            # Convert RGBA to RGB
            background = Image.new('RGB', img.size, (255, 255, 255)) # white background
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            img = background
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

# Update the generate_data_json function to include raw scores
def generate_data_json(winoground_dataset, model_scores):
    data = []
    for idx, example in tqdm(enumerate(winoground_dataset.winoground), total=len(winoground_dataset.winoground), desc="Processing examples"):
        image_0_path = os.path.join(winoground_dataset.root_dir, winoground_dataset.metadata[idx]['image_0'])
        image_1_path = os.path.join(winoground_dataset.root_dir, winoground_dataset.metadata[idx]['image_1'])
        example_data = {
            "id": idx,
            "image_0": image_to_base64(image_0_path),
            "image_1": image_to_base64(image_1_path),
            "caption_0": example['caption_0'],
            "caption_1": example['caption_1'],
            "scores": {},
            "tags": {
                "original": [tag for tag, ids in winoground_dataset.original_tags.items() if idx in ids],
                "new": [tag for tag, ids in winoground_dataset.new_tags.items() if idx in ids]
            }
        }
        for model_name, scores in model_scores.items():
            score = scores[idx]
            example_data["scores"][model_name] = {
                "text": int(text_correct(score)),
                "image": int(image_correct(score)),
                "group": int(group_correct(score)),
                "raw": {
                    "c0_i0": score["c0_i0"].item(),
                    "c0_i1": score["c0_i1"].item(),
                    "c1_i0": score["c1_i0"].item(),
                    "c1_i1": score["c1_i1"].item()
                }
            }
        data.append(example_data)
    return json.dumps(data)

def generate_html_app(data_json):
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Winoground Visualization</title>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <style>
        .example {{ border: 1px solid #ddd; margin: 10px; padding: 10px; }}
        .images {{ display: flex; justify-content: space-around; }}
        .image {{ text-align: center; }}
        .image img {{ max-width: 500px; max-height: 500px; width: auto; height: auto; }}
        .scores {{ margin-top: 10px; }}
        .filters {{ margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .example-count {{ margin-top: 10px; font-weight: bold; }}
        .tag-filter {{ margin-top: 10px; }}
    </style>
</head>
<body>
    <div id="app">
        <div class="filters">
            <div v-for="(filter, index) in filters" :key="index">
                <label>Model:</label>
                <select v-model="filter.model">
                    <option value="">All models</option>
                    <option v-for="model in modelNames" :value="model">{{{{ model }}}}</option>
                </select>
                <label>Score:</label>
                <select v-model="filter.score">
                    <option value="">All scores</option>
                    <option value="text">Text</option>
                    <option value="image">Image</option>
                    <option value="group">Group</option>
                </select>
                <label>Correctness:</label>
                <select v-model="filter.correctness">
                    <option value="">All</option>
                    <option value="correct">Correct</option>
                    <option value="incorrect">Incorrect</option>
                </select>
                <button @click="removeFilter(index)">Remove Filter</button>
            </div>
            <button @click="addFilter">Add Filter</button>
        </div>
        <div class="tag-filter">
            <label>Original Tag:</label>
            <select v-model="selectedOriginalTag">
                <option value="">All</option>
                <option v-for="tag in originalTags" :value="tag">{{{{ tag }}}}</option>
            </select>
            <label>New Tag:</label>
            <select v-model="selectedNewTag">
                <option value="">All</option>
                <option v-for="tag in newTags" :value="tag">{{{{ tag }}}}</option>
            </select>
        </div>
        <div>
            <label>
                <input type="checkbox" v-model="randomizeOrder"> Randomize order
            </label>
        </div>
        <div class="example-count">
            Displaying {{{{ displayedExamples.length }}}} out of {{{{ filteredExamples.length }}}} examples
        </div>
        <div v-for="example in displayedExamples" :key="example.id" class="example">
            <h3>Example {{{{ example.id + 1 }}}}</h3>
            <div class="images">
                <div class="image">
                    <img :src="'data:image/png;base64,' + example.image_0">
                    <p>{{{{ example.caption_0 }}}}</p>
                </div>
                <div class="image">
                    <img :src="'data:image/png;base64,' + example.image_1">
                    <p>{{{{ example.caption_1 }}}}</p>
                </div>
            </div>
            <div class="scores">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Text</th>
                            <th>Image</th>
                            <th>Group</th>
                            <th>c0_i0</th>
                            <th>c0_i1</th>
                            <th>c1_i0</th>
                            <th>c1_i1</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="(scores, model) in example.scores" :key="model">
                            <td>{{{{ model }}}}</td>
                            <td>{{{{ scores.text }}}}</td>
                            <td>{{{{ scores.image }}}}</td>
                            <td>{{{{ scores.group }}}}</td>
                            <td>{{{{ scores.raw.c0_i0.toFixed(4) }}}}</td>
                            <td>{{{{ scores.raw.c0_i1.toFixed(4) }}}}</td>
                            <td>{{{{ scores.raw.c1_i0.toFixed(4) }}}}</td>
                            <td>{{{{ scores.raw.c1_i1.toFixed(4) }}}}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div>
                <strong>Original Tags:</strong> {{{{ example.tags.original.join(', ') }}}}
            </div>
            <div>
                <strong>New Tags:</strong> {{{{ example.tags.new.join(', ') }}}}
            </div>
        </div>
    </div>
    <script>
        const app = Vue.createApp({{
            data() {{
                return {{
                    examples: {0},
                    filters: [],
                    randomizeOrder: false,
                    selectedOriginalTag: '',
                    selectedNewTag: ''
                }}
            }},
            computed: {{
                modelNames() {{
                    return Object.keys(this.examples[0].scores);
                }},
                originalTags() {{
                    return [...new Set(this.examples.flatMap(ex => ex.tags.original))].sort();
                }},
                newTags() {{
                    return [...new Set(this.examples.flatMap(ex => ex.tags.new))].sort();
                }},
                filteredExamples() {{
                    return this.examples.filter(example => {{
                        const passesOriginalTag = !this.selectedOriginalTag || example.tags.original.includes(this.selectedOriginalTag);
                        const passesNewTag = !this.selectedNewTag || example.tags.new.includes(this.selectedNewTag);
                        return passesOriginalTag && passesNewTag;
                    }});
                }},
                displayedExamples() {{
                    let examples = this.filteredExamples.filter(example => {{
                        return this.filters.every(filter => {{
                            if (filter.model && filter.score && filter.correctness) {{
                                const score = example.scores[filter.model][filter.score];
                                return filter.correctness === 'correct' ? score === 1 : score === 0;
                            }}
                            return true;
                        }});
                    }});
                    
                    if (this.randomizeOrder) {{
                        return [...examples].sort(() => Math.random() - 0.5);
                    }}
                    return examples;
                }}
            }},
            methods: {{
                addFilter() {{
                    this.filters.push({{ model: '', score: '', correctness: '' }});
                }},
                removeFilter(index) {{
                    this.filters.splice(index, 1);
                }}
            }}
        }}).mount('#app');
    </script>
</body>
</html>
    """.format(data_json)

if __name__ == '__main__':
    model_names = [
        'llava-v1.5-13b',
        'train_sugarcrepe_only_swap_lora2',
        # 'train_sugarcrepe_combined_only_swap_lora2',
        'train_sugarcrepe_combined_only_swap_labsmooth_0.1',
        # 'train_sugarcrepe_combined_only_swap_hinge_loss',
        # 'train_my_sugarcrepe_neg_cot_filled_combined',
        # 'train_my_sugarcrepe_neg_cot_filled',
        
        # 'train_coco_syn_swap_v1',
        # 'train_coco_syn_cot_adv_ref',
        'train_sugarcrepe_combined_only_swap_labsmooth_0.1_rmyesno_eos',
        'train_sugarcrepe_combined_only_swap_labsmooth_0.1_logprob_avg',
        'train_sugarcrepe_combined_w_explanation_only_swap_labsmooth_0.1',
        
        'train_coco_syn_cot_adv_ref',
        
        'train_coco_syn_cot_adv_ref_combined_labsmooth_0.1_avg_logprob_best_ckpt',
        'train_coco_syn_cot_adv_ref_combined_labsmooth_0.1_avg_logprob',
        
        'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1_avg_logprob_best_ckpt',
        'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1_avg_logprob',
    ]
    
    winoground_root_dir = '/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets'
    winoground_dataset = Winoground(root_dir=winoground_root_dir)
    
    model_scores = {}
    for model_name in model_names:
        scores_file = f'playground/data/eval/winoground/eval_caption_{model_name}/scores.pt'
        scores = torch.load(scores_file, map_location=torch.device('cpu'))
        model_scores[model_name] = get_winoground_scores(scores)
    
    data_json = generate_data_json(winoground_dataset, model_scores)
    html_content = generate_html_app(data_json)
    
    output_path = 'playground/dev/winoground_visualization_app_caption_scores.html'
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"HTML app generated at {output_path}")