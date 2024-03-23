import os
import textwrap
import yaml
import argparse

from parcellate.cfg import *
from parcellate.util import CFG_FILENAME, join, get_action
from parcellate.model import parcellate

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Compute a subject-specific brain parcellation.')
    argparser.add_argument('config_path')
    argparser.add_argument('-p', '--parcellation_id', default=None, help=textwrap.dedent('''\
        ID (name) of parcellation configuration to use for setting the output directory for any parcellation or
        aggregation steps outside of the grid search inner loop. If ``None``, uses the first parcellation setting. \
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
    config_path = args.config_path
    parcellation_id = args.parcellation_id
    nogrid = args.nogrid
    overwrite = args.overwrite

    cfg = get_cfg(config_path)

    action_sequence = get_action_sequence(
        cfg,
        'parcellate',
        parcellation_id
    )

    assert len(action_sequence) >= 3, ('Dependency configuration error. A parcellation requires at least 3 actions: '
        'sample, align, and parcellate. Got the following value of deps: %s.' % action_sequence)

    sample_id = get_action('sample', action_sequence)['id']
    alignment_id = get_action('align', action_sequence)['id']
    evaluation_id = get_action('evaluate', action_sequence)['id']
    aggregation_id = get_action('aggregate', action_sequence)['id']
    parcellation_id = get_action('parcellate', action_sequence)['id']

    # Parcellation's predecessor is always the 1st entry of the deps
    parcellation_predecessor = action_sequence[1]['type']
    parcellation_predecessor_id = action_sequence[1]['id']

    assert not parcellation_predecessor is None, 'Dependency error. No dependency found for parcellation'

    sample_kwargs = get_kwargs(cfg, 'sample', sample_id)
    align_kwargs = get_kwargs(cfg, 'align', alignment_id)
    evaluate_kwargs = get_kwargs(cfg, 'evaluate', evaluation_id)
    aggregate_kwargs = get_kwargs(cfg, 'aggregate', aggregation_id)

    output_dir = cfg.get('output_dir', None)
    compress_outputs = cfg.get('compress_outputs', True)

    assert output_dir is not None, '``output_dir`` must be provided in cfg. Terminating.'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, CFG_FILENAME), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if aggregate_kwargs and 'grid' in cfg and not nogrid:
        grid_params = get_grid_params(cfg)
    else:
        grid_params = None

    parcellate(
        output_dir,
        action_sequence,
        sample_kwargs,
        align_kwargs,
        evaluate_kwargs=evaluate_kwargs,
        aggregate_kwargs=aggregate_kwargs,
        grid_params=grid_params,
        compress_outputs=compress_outputs,
        overwrite=overwrite
    )
