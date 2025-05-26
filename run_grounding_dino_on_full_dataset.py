import os
import sys
import json
import pandas as pd
import random
from tqdm import tqdm
import torch
from PIL import Image
#print('importing transformers and ultralytics...')
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLOWorld
#print('done importing transformers and ultralytics')
from get_dino_queries import extract_noun_phrases_with_counts
from detection_plotter import draw_detections


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/imgeneval-data')
DEVICE = 'cuda'
DETECTOR_TYPE = 'GroundingDINO' #other option is YOLO-World
SAVE_FREQ = 5000
DATA_STRIDE = 36
BAD_IMAGE_BASES = ['0f35f2c329da2186.jpg', '1efbcf592fa4d655.jpg', '433a2130ab98a8ef.jpg', '827ee98b329837db.jpg', '69bf095cf5a40fb9.jpg', '7bee38652a9b7022.jpg', '804dd1c559ded79f.jpg', '96818368e36b4fb8.jpg', 'bf6ad39c9d039592.jpg', 'image_path', 'eefb23c44a78d813.jpg']


def load_caption_dict():
    print('loading caption dict...')
    caption_dict = {}
    df = pd.read_csv(os.path.join(BASE_DIR, 'OpenImages/train.csv'), header=None, names=['image_path', 'positive_caption', 'negative_caption'])
    for row in df.itertuples(index=False):
        caption_dict[os.path.basename(row.image_path)] = row.positive_caption

    print('done loading caption dict')
    return caption_dict


def get_output_dict_filename(threshold, offset):
    if DETECTOR_TYPE == 'GroundingDINO':
        return os.path.join(BASE_DIR, 'groundingdino_threshold%s_shard%d.json'%(str(threshold), offset))
    elif DETECTOR_TYPE == 'YOLO-World':
        return os.path.join(BASE_DIR, 'yoloworld_threshold%s_shard%d.json'%(str(threshold), offset))
    else:
        assert(False)


def setup_tools():
    print('setting up tools...')
    if DETECTOR_TYPE == 'GroundingDINO':
        model_id = 'IDEA-Research/grounding-dino-base'
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
        tools = {'processor' : processor, 'model' : model}
    elif DETECTOR_TYPE == 'YOLO-World':
        model = YOLOWorld("yolov8x-worldv2.pt")
        tools = {'model' : model}
    else:
        assert(False)

    print('done setting up tools')
    return tools


#will return "results" from tutorial
def run_grounding_dino_one_image(image_filename, classnames, tools, threshold):
    if DETECTOR_TYPE == 'GroundingDINO':
        image = Image.open(image_filename).convert('RGB')
        inputs = tools['processor'](images=image, text=[classnames], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs = tools['model'](**inputs)

        detection_results = tools['processor'].post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=threshold, target_sizes=[image.size[::-1]])
        assert(len(detection_results) == 1)
        detection_results = detection_results[0]
        detection_results['boxes'] = detection_results['boxes'].cpu().numpy().tolist()
        detection_results['scores'] = detection_results['scores'].cpu().numpy().tolist()
    elif DETECTOR_TYPE == 'YOLO-World':
        tools['model'].set_classes(classnames)
        results = tools['model'].predict(image_filename, conf=threshold)
        detection_results = {}
        detection_results['boxes'] = results[0].boxes.xyxy.cpu().numpy().tolist()
        detection_results['scores'] = results[0].boxes.conf.cpu().numpy().tolist()
        detection_results['text_labels'] = [classnames[int(idx)] for idx in results[0].boxes.cls.cpu().numpy().tolist()]
    else:
        assert(False)

    return detection_results


#def visualize(image_filename, detection_results, threshold):
#    image = Image.open(image_filename).convert('RGB')
#    detections = {}
#    for box, label in zip(detection_results['boxes'], detection_results['text_labels']):
#        if label not in detections:
#            detections[label] = []
#
#        detections[label].append(box)
#
#    detector_type = {'GroundingDINO' : 'groundingdino', 'YOLO-World' : 'yoloworld'}[DETECTOR_TYPE]
#    plot_filename = os.path.join(BASE_DIR, '%s_threshold%s_vis'%(detector_type, str(threshold)), os.path.basename(image_filename))
#    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
#    draw_detections(image, detections, plot_filename)


def run_grounding_dino_on_full_dataset(threshold, offset):
    threshold = float(threshold)
    offset = int(offset)

    tools = setup_tools()
    caption_dict = load_caption_dict()
    image_bases = sorted(caption_dict.keys())
    image_bases = image_bases[offset::DATA_STRIDE]
    output_dict_filename = get_output_dict_filename(threshold, offset)
    output_dict = {}
    if os.path.exists(output_dict_filename):
        with open(output_dict_filename, 'r') as f:
            output_dict = json.load(f)

        print('oh goody! already did %d examples!'%(len(output_dict)))

    image_bases = [image_base for image_base in image_bases if image_base not in output_dict]
    t = 0
    for image_base in tqdm(image_bases):
        if os.path.exists('kill%d'%(offset)):
            print('apoptosis %d'%(offset))
            assert(False)

        if image_base in BAD_IMAGE_BASES:
            print('!!')
            continue

        t += 1
        caption = caption_dict[image_base]
        parser_results = extract_noun_phrases_with_counts(caption)
        classnames = sorted(parser_results.keys())
        image_filename = os.path.join(BASE_DIR, 'OpenImages/images/train', image_base)
        print('MEOW!!! ' + image_base)
        detection_results = run_grounding_dino_one_image(image_filename, classnames, tools, threshold)
        output = {'caption' : caption, 'parser_results' : parser_results, 'detection_results' : detection_results}
        output_dict[image_base] = output
        if t > 0 and t % SAVE_FREQ == 0:
            with open(output_dict_filename, 'w') as f:
                json.dump(output_dict, f)

    with open(output_dict_filename, 'w') as f:
        json.dump(output_dict, f)


def usage():
    print('Usage: python run_grounding_dino_on_full_dataset.py <threshold> <offset>')


if __name__ == '__main__':
    run_grounding_dino_on_full_dataset(*(sys.argv[1:]))
