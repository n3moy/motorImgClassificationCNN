import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser(
        description='NN pipeline'
    )
    parser.add_argument(
        '--config',
        type=str, default=Path(__file__).parent.parent / 'configs' / 'model' / 'train.yaml',
        help='Config path'
    )
    parser.add_argument(
        '--mode',
        type=str, default='train',
        choices=['train', 'test', 'inference'],
        help='Run mode'
    )
    parser.add_argument(
        '--augmented',
        type=bool, default=False,
        help='Use augmented data for train'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config_path = args.config
    config = yaml.load(config_path)
    mode = args.mode
    
    time = datetime.today().date().isoformat()
    log_filename = f'logging_{time}.log'
    log_path = Path(__file__) / 'logs' / mode / log_filename
    logging.basicConfig(
        filename=log_path, 
        level=logging.DEBUG, 
        format='%(asctime)s %(levelname)s: %(message)s',
        filemode='a+'
    )
    # TODO Add functions to run modes

