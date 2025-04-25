import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import numpy as np
from openai import OpenAI

import json
from pathlib import Path


PROMPT1 = """
I need more examples of pairs of captions for images. 

Here are some rules :
1. Both captions in a pair should have roughly the same set of words.
2. The two captions should differ in the ordering of words such that they describe visual concepts that are different. 
3. The two captions should be grammatical.
4. Make sure the two captions make sense as captions for images. They should be logical. Make sure that the two captions describe different images. Pay particular attention to this.

Here are some examples :
```
{captions}
```
Please provide {num_captions} more examples.
"""

PROMPT2 = """
Can you double check which of these pairs might simply be paraphrases and not refer to distinct visual scenes? If so can you correct one of the captions with minimal editing? Also make the edit if one of the captions is non-sensical.
"""

PROMPT3 = """
After correction, please output only the final set of {num_captions} caption pairs in json format.
"""

def parse_args(ret='parsed'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to use')
    parser.add_argument('--infile', type=str, default='data/openai/winoground_incontext_examples/combined_4seeds_15eg.json', help='In context examples file')
    parser.add_argument('--num_ic_eg', type=int, default=10, help='Number of in context examples to use')
    parser.add_argument('--num_captions', type=int, default=10, help='Number of caption pairs to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--outfile', type=str, default='output_captions', help='Output file prefix')
    parser.add_argument('--outpath', type=str, default='data/test/openai', help='Output path')
    
    if ret == 'parsed':
        return parser.parse_args()
    elif ret == 'default':
        return parser.parse_args([])
    elif ret == 'parser':
        return parser
    else:
        raise ValueError(f'Invalid return type: {ret}')

    

def main():
    args = parse_args('parsed')
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'], max_retries=3, timeout=50.0)
    
    captions = json.load(open(args.infile, 'r'))
    captions = [{'captionA' : caption['captionA'], 'captionB' : caption['captionB']} for caption in captions] # getting rid of winoground id for this
    
    if len(captions) > args.num_ic_eg:
        RNG = np.random.RandomState(args.seed)
        captions = list(RNG.choice(captions, args.num_ic_eg, replace=False))
    
    # TODO : Try providing a system message
    messages = [
        {
            'role' : 'user',
            'content' : PROMPT1.format(captions=json.dumps(captions), num_captions=args.num_captions)
        }
    ]
    chat_completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=1000,
        seed=args.seed,
    )
    print('Got response1')
    
    # resp1 is the initial set of caption pairs from the model
    resp1 = chat_completion.choices[0].message
    messages.append(dict(resp1))
    messages.append({
        'role' : 'user',
        'content' : PROMPT2
    })
    chat_completion2 = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=1000,
        seed=args.seed,
    )
    print('Got response2')
    
    # resp2 is corrections from the model
    resp2 = chat_completion2.choices[0].message
    messages.append(dict(resp2))
    messages.append({
        'role' : 'user',
        'content' : PROMPT3.format(num_captions=args.num_captions)
    })
    chat_completion3 = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=1000,
        seed=args.seed,
        response_format={
            'type' : 'json_object'
        }
    )
    
    # resp3 is the final set of caption pairs from the model
    resp3 = chat_completion3.choices[0].message
    print('Got response3')
    messages.append(dict(resp3))
    
    if not os.path.exists(Path(args.outpath)):
        os.makedirs(Path(args.outpath))
        
    with open(Path(args.outpath) / f'{args.outfile}_all_messages.json', 'w') as f:
        json.dump(messages, f, indent=4)
        
    # parse this into json
    output_captions = json.loads(resp3.content)
    with open(Path(args.outpath) / f'{args.outfile}.json', 'w') as f:
        json.dump(output_captions, f, indent=4)
    
if __name__ == '__main__':
    main()

