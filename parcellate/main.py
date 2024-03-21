import os
import textwrap
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import argparse

from parcellate.cfg import *
from parcellate.util import CFG_FILENAME, join
from parcellate.model import run, run_grid

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Compute a subject-specific brain parcellation.')
    argparser.add_argument('config_path')
    argparser.add_argument('-m', '--mode', nargs='+', default=['all'], help=textwrap.dedent('''\
        Step to run. Some set of ``parcellate``, ``align``, ``evaluate``, ``aggregate``, or ``all`` (runs all steps in 
        sequence). If ``grid`` is specified in the config, apply the desired step(s) to the entire grid and output 
        results to subdirectory ``grid``. Otherwise, apply the desired step(s) to a single model setting and output 
        results to subdirectory ``parcellation_<PARCELLATION_ID>`` (see CLI arg ``parcellation_id``). If ``grid`` is 
        used, the best entry in the grid (highest average correlation to reference atlas(es)) will either be copied to 
        subdirectory ``parcellation_<EVALUATION_ID>``, or (if ``refit`` is specified in the config) refitted using the  
        best parameters, with results output to subdirectory ``parcellation_<EVALUATION_ID>``. Defaults to ``'all'``.\
        '''
    ))
    argparser.add_argument('-p', '--parcellation_id', default='main', help=textwrap.dedent('''\
        ID (name) of parcellation configuration to use for setting the output directory for any parcellation or
        aggregation steps outside of the grid search inner loop. Defaults to ``main``. \
        '''
    ))
    argparser.add_argument('-a', '--alignment_id', default='main', help=textwrap.dedent('''\
        ID (name) of alignment configuration to use for any required alignment steps. Defaults to ``main``. \
        '''
    ))
    argparser.add_argument('-e', '--evaluation_id', default='main', help=textwrap.dedent('''\
        ID (name) of evaluation configuration to use for any required evaluation steps. Defaults to ``main``.\
        '''
    ))
    argparser.add_argument('-g', '--aggregation_id', default='main', help=textwrap.dedent('''\
        ID (name) of aggregation configuration to use for any required aggregation steps. Used only for grid searches 
        (ignored for individual runs). Defaults to ``main``.\
        '''
    ))
    argparser.add_argument('-n', '--nogrid', action='store_true', help=textwrap.dedent('''\
        Do not grid search, even if ``grid`` is provided.\
        '''
    ))
    argparser.add_argument('-o', '--overwrite', action='store_true', help=textwrap.dedent('''\
        Whether to overwrite existing parcellation data. If ``False``, will only estimate missing results, leaving old 
        ones in place. Note that, for safety reasons, parcellate never deletes files or directories, so ``overwrite``
        will clobber any existing files that need to be rebuilt, but it will leave any other older files in place.
        As a result, the output directory may contain a mix of old and new files. To avoid this, you must delete
        existing directories yourself.\
        '''
    ))
    args = argparser.parse_args()
    mode = set(args.mode)
    cfg = get_cfg(args.config_path)

    parcellate_kwargs = align_kwargs = evaluate_kwargs = aggregate_kwargs = refit_kwargs = None
    if mode & {'parcellate', 'all'}:
        parcellate_kwargs = get_parcellate_kwargs(cfg)
    if mode & {'align', 'all'}:
        align_kwargs = get_align_kwargs(cfg, args.alignment_id)
    if mode & {'evaluate', 'all'}:
        evaluate_kwargs = get_evaluate_kwargs(cfg, args.evaluation_id)
    if mode & {'aggregate', 'all'}:
        aggregate_kwargs = get_aggregate_kwargs(cfg, args.evaluation_id)
    if mode & {'refit', 'all'}:
        refit_kwargs = get_refit_kwargs(cfg)

    output_dir = cfg.get('output_dir', None)
    compress_outputs = cfg.get('compress_outputs', True)

    assert output_dir is not None, '``output_dir`` must be provided in cfg. Terminating.'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, CFG_FILENAME), 'w') as f:
        yaml.dump(cfg, f)

    if 'grid' in cfg and not args.nogrid:
        grid_params = get_grid_params(cfg)
        run_grid(
            parcellate_kwargs=parcellate_kwargs,
            align_kwargs=align_kwargs,
            evaluate_kwargs=evaluate_kwargs,
            aggregate_kwargs=aggregate_kwargs,
            refit_kwargs=refit_kwargs,
            grid_params=grid_params,
            parcellation_id=args.parcellation_id,
            alignment_id=args.alignment_id,
            evaluation_id=args.evaluation_id,
            aggregation_id=args.aggregation_id,
            output_dir=output_dir,
            overwrite=args.overwrite
        )
    elif (
            parcellate_kwargs is not None or
            align_kwargs is not None or
            evaluate_kwargs is not None
    ):
        run(
            parcellate_kwargs=parcellate_kwargs,
            align_kwargs=align_kwargs,
            evaluate_kwargs=evaluate_kwargs,
            parcellation_id=args.parcellation_id,
            alignment_id=args.alignment_id,
            evaluation_id=args.evaluation_id,
            output_dir=output_dir,
            overwrite=args.overwrite
        )
    else:
        print('Nothing to do. Terminating.')
