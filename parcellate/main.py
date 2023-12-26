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
    args = argparser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = yaml.load(f, Loader=Loader)

    parcellate(**cfg)