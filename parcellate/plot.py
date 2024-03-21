import os
import pandas as pd
from matplotlib import pyplot as plt
import textwrap
import argparse

from parcellate.cfg import *
from parcellate.util import *


def plot_grids(
        cfg_paths,
        dimensions=None,
        reference_atlases=None,
        evaluation_atlases=None,
        aggregation_id='main',
        output_dir='parcellate_plots'
):
    if dimensions is None:
        dimensions = []
    if reference_atlases is None:
        reference_atlases = []
    if evaluation_atlases is None:
        evaluation_atlases = {}
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]
    for cfg_path in cfg_paths:
        cfg = get_cfg(cfg_path)
        output_dir = cfg['output_dir']
        compressed = cfg.get('compress_outputs', True)
        df_path = get_aggregation_path(output_dir, aggregation_id=aggregation_id)
        df = pd.read_csv(df_path)
        print(df)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot parcellation performance')
    argparser.add_argument('cfg_paths', nargs='+', help=textwrap.dedent('''\
        Path(s) to ``parcellate`` config files (config.yml) to plot.'''
    ))
    argparser.add_argument('reference_atlases', default=None, help=textwrap.dedent('''\
        Name of reference atlas(es) to use for plotting. If ``None``, use all available reference atlases.'''
    ))
    argparser.add_argument('evaluation_atlases', default=None, help=textwrap.dedent('''\
        Name of evaluation atlas(es) to use for plotting. If ``None``, use all available evaluation atlases.'''
    ))
    argparser.add_argument('aggregation_id', default='main', help=textwrap.dedent('''\
        Value of ``aggregation_id`` from which to extract grid search performance data. Defaults to ``main``.'''
    ))
    argparser.add_argument('dimensions', nargs='+', default=None, help=textwrap.dedent('''\
        Name of grid-searched dimension(s) to plot. If ``None``, use all available dimensions.'''
    ))
    args = argparser.parse_args()

    cfg_paths = args.cfg_paths
    reference_atlases = args.reference_atlases
    evaluation_atlases = args.evaluation_atlases
    aggregation_id = args.aggregation_id
    dimensions = args.dimensions