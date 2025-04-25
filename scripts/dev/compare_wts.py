import os
import sys
sys.path.append(os.path.abspath('.'))

from llava.model.eval_model.gptscore_model import LLaVA_GPTScoreModel
import torch

if __name__ == '__main__':
    model2 = LLaVA_GPTScoreModel(
        model_path='checkpoint/output/winoground_dpo_train2/', model_name='llava-v1.5-13b-lora-winoground2', 
        model_base='liuhaotian/llava-v1.5-13b', device='cpu')
    model1 = LLaVA_GPTScoreModel(model_path='liuhaotian/llava-v1.5-13b', model_name='llava-v1.5-13b', device='cpu')

    for (n1, p1), (n2, p2) in zip(model1.model.named_parameters(), model2.model.named_parameters()):
        if n1 != n2:
            print('Names differ : ', n1, n2)
        elif not torch.allclose(p1, p2):
            print('Values differ : ', n1)
            print(torch.norm(p1 - p2))

