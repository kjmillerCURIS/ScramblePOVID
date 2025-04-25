import os
import json
import argparse
import pandas as pd
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--add-video", action='store_true', default=False)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-upload-file", type=str)
    parser.add_argument("--output-csv-file", type=str)
    parser.add_argument("--ckpt_name", type=str)
    return parser.parse_args()

# llama3_phrases = ['answer is', 'option is']

def compare_opts(res, gt):
    if res.upper().startswith(gt.upper()):
        return True
    # elif any(phrase in res.lower() for phrase in llama3_phrases):
    #     for phrase in llama3_phrases:
    #         if phrase in res.lower():
    #             res = res.split(phrase)[1].strip()
    #             if res.upper().startswith(gt.upper()):
    #                 return True
    return False

def eval_single(result_file, eval_only_type=None):
    results = {}
    accs = {}
    for line in open(result_file):
        row = json.loads(line)
        results[row['question_id']] = row

    type_counts = {}
    correct_counts = {}
    for row_idx, question_data in data.iterrows():
        if eval_only_type is not None and question_data['q_ind'] != eval_only_type: continue
        data_type = question_data['q_ind']
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        try:
            question_id = int(question_data['id'])
        except:
            question_id = question_data['id']
        if question_id not in results:
            correct_counts[data_type] = correct_counts.get(data_type, 0)
            continue
        row = results[question_id]
        # if row['text'] == question_data['answer']:
        if compare_opts(row['text'], question_data['correct']):
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

    total_count = 0
    total_correct = 0
    for data_type in sorted(type_counts.keys()):
        accuracy = correct_counts[data_type] / type_counts[data_type] * 100
        if eval_only_type is None:
            # print(f"{ques_type_id_to_name[data_type]}: {accuracy:.2f}%")
            # accs[ques_type_id_to_name[data_type]] = accuracy
            print(f"question type {data_type}: {accuracy:.2f}%")
            accs[f"question type {data_type}"] = accuracy

        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    total_accuracy = total_correct / total_count * 100
    if eval_only_type is None:
        print(f"Total accuracy: {total_accuracy:.2f}%")
        accs['total'] = total_accuracy
    else:
        print(f"question type {eval_only_type} accuracy: {total_accuracy:.2f}%")
        accs[f"question type {eval_only_type}"] = total_accuracy

    return results, accs

def eval_overall(result_file):
    results = {}
    for line in open(result_file):
        row = json.loads(line)
        results[row['question_id']] = row
        
    total_count = 0
    total_correct = 0
    for row_idx, question_data in data.iterrows():
        qid = row_idx
        if compare_opts(results[qid]['text'], question_data['answer']):
            total_correct += 1
        total_count += 1
    return results, {'total' : total_correct / total_count * 100}

if __name__ == "__main__":
    # NOTE : variables in "main" are global
    args = get_args()
    data = pd.read_csv(args.annotation_file)
    # ques_type_id_to_name = {id:n for n,id in data['question_type'].items()}

    result_dict = {'model': args.ckpt_name}
    if 'HUMAN_FILTER' in Path(args.annotation_file).name:
        results, accs = eval_single(args.result_file)
    else:
        results, accs = eval_overall(args.result_file)
    result_dict.update(accs)
    df = pd.DataFrame(result_dict, index=[0])
    
    os.makedirs(os.path.dirname(args.output_csv_file), exist_ok=True)
    df.to_csv(args.output_csv_file, index=False)
    # eval_single(args.result_file, eval_only_type='image')
    # if args.add_video:
    #     eval_single(args.result_file, eval_only_type='video')
        
    # TODO : output results into maybe a csv instead of simply printing
    if 'HUMAN_FILTER' in Path(args.annotation_file).name:
        os.makedirs(os.path.dirname(args.result_upload_file), exist_ok=True)
        with open(args.result_upload_file, 'w') as fp:
            for row_idx, question in data.iterrows():
                qid = question['id']
                if qid in results:
                    result = results[qid]
                else:
                    result = results[int(qid)]
                fp.write(json.dumps({
                    'question_id': qid,
                    'prediction': result['text']
                }) + '\n')
