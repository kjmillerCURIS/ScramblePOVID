from gradio_client import Client, handle_file
import argparse
import os
import shutil
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, help='Path to the file to evaluate', default='playground/data/eval/mm-vet/results/llava-v1.5-13b.json')
parser.add_argument('--result_path', type=str, help='Path to save the result', default='')
args = parser.parse_args()


client = Client("whyu/MM-Vet_Evaluator")
result = client.predict(
    file_obj=handle_file(args.file_path),
    api_name="/grade",
)
print(result) # filename

if not args.result_path:
    args.result_path = Path(args.file_path).parents[1] / 'gradio_out' / Path(args.file_path).name.replace('.json', '.zip')
    
os.makedirs(Path(args.result_path).parent, exist_ok=True)
shutil.copy(result, args.result_path)

with zipfile.ZipFile(args.result_path, 'r') as zip_ref:
    zip_ref.extractall(Path(args.result_path).parent)
