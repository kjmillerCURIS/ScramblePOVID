import os
import sys
from apply_groundingdino_strategy import apply_groundingdino_strategy


if __name__ == '__main__':
    for strategy in ['check_presence', 'check_singular_and_plural', 'check_thorough']:
        for threshold in [0.05, 0.15]:
            apply_groundingdino_strategy(threshold, strategy)
