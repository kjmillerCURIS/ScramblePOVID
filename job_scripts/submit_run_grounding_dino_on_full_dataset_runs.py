import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_run_grounding_dino_on_full_dataset.sh'
THRESHOLD = -1.0
OFFSETS = range(36)


def submit_run_grounding_dino_on_full_dataset_runs():
    for offset in OFFSETS:
        job_name = 'groundingdinofull_%s_%d'%(str(THRESHOLD), offset)
        my_cmd = 'qsub -N %s -v THRESHOLD=%s,OFFSET=%d %s'%(job_name, str(THRESHOLD), offset, SCRIPT_NAME)
        print('submitting training run: "%s"'%(my_cmd))
        os.system(my_cmd)
        if DEBUG:
            print('DEBUG MODE: let\'s see how that first run goes...')
            return


if __name__ == '__main__':
    submit_run_grounding_dino_on_full_dataset_runs()
