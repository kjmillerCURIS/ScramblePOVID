import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    filelist_root_dir = Path('playground/data/eval/conme_orig/')
    new_filelist_root_dir = Path('playground/data/eval/conme/')
    new_filelist_root_dir.mkdir(parents=True, exist_ok=True)
    my_data_root = Path('/projectnb/ivc-ml/samarth/datasets/COCO/images/val2017')
    for file in filelist_root_dir.glob('*.csv'):
        if 'HUMAN_FILTER' in file.name:
            continue
        df = pd.read_csv(file)
        # df['image'] = df['image'].apply(lambda x: f'{Path(x).parent.name}/{Path(x).name}')
        df['image'] = df['image'].apply(lambda x: f'val2017/{Path(x).name}')
        df.to_csv(new_filelist_root_dir / file.name, index=False)
        print(f'Fixed {file}')
    print('Done')