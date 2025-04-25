import os
import sys
sys.path.append(os.path.abspath('.'))

import subprocess

if __name__ == '__main__':
    for i in range(21, 22):
        command = ['python', 'openai/get_captions.py']
        command += ['--model', 'gpt-4o']
        command += ['--infile', 'data/openai/winoground_incontext_examples/combined_4seeds_15eg.json']
        command += ['--outpath', 'data/openai/generated_caption_pairs']
        command += ['--outfile', f'gpt4o_examples_seed{i}'] # only the prefix
        command += ['--num_ic_eg', '10']
        command += ['--seed', str(i)]
        print(command)
        subprocess.run(command)
        print(f'Finished running {i}')