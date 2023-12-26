import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import argparse

from parcellate.model import parcellate

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Compute a subject-specific brain parcellation.')
    argparser.add_argument('config_path')
    argparser.add_argument('-o', '--overwrite', action='store_true', help='Whether to overwrite existing ' + \
                           'parcellation data. If ``False``, will only estimate missing results, leaving old ' + \
                           'ones in place.')
    args = argparser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = yaml.load(f, Loader=Loader)

    parcellate(overwrite=args.overwrite, **cfg)