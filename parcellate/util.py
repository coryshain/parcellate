import os
import itertools
import numpy as np

from parcellate.constants import *


def process_grid_params(grid_params):
    _grid_params = {}
    if grid_params:
        for grid_param in grid_params:
            vals = grid_params[grid_param]
            _vals = []
            for val in vals:
                if isinstance(val, list):
                    _val = []
                    for v in val:
                        vi = int(v)
                        vf = float(v)
                        if vi == vf :  # Integer
                            _val.append(vi)
                        else:
                            _val.append(vf)
                    __vals = np.arange(*_val).tolist()
                else:
                    __vals = [val]
                _vals.extend(__vals)
            _grid_params[grid_param] = _vals
    else:
        _grid_params = {'model': [0]}
    keys = sorted(list(_grid_params.keys()))
    cross = itertools.product(*[_grid_params[x] for x in keys])

    for x in cross:
        row = {}
        for _x, key in zip(x, keys):
            row[key] = _x

        yield row


def get_action_id(
        cfg,
        id_type,
        action_id=None
):
    _cfg = cfg.get(id_type, {})
    if action_id is None:
        keys = list(_cfg.keys())
        if not keys:
            action_id = 'main'
        else:
            action_id = keys[0]
    else:
        assert not _cfg or action_id in _cfg, '%s ID %s not found in config' % (id_type, action_id)

    return action_id

def process_action_ids(
        cfg,
        parcellation_id=None,
        alignment_id=None,
        evaluation_id=None,
        aggregation_id=None
):
    # Aggregate
    aggregation_id = get_action_id(cfg, 'aggregate', aggregation_id)

    # Evaluate
    _evaluation_id = cfg.get('aggregate', {}).get(aggregation_id, {}).get('evaluation_id', None)
    if _evaluation_id:
        assert evaluation_id is None or _evaluation_id == evaluation_id, (
                'Mismatch between requested ``evaluation_id`` (%s) and the one required by aggregation_id %s (%s).' %
                (evaluation_id, aggregation_id, _evaluation_id)
        )
        evaluation_id = _evaluation_id
    else:
        evaluation_id = get_action_id(cfg, 'evaluate', evaluation_id)

    # Align
    _alignment_id = cfg.get('evaluate', {}).get(evaluation_id, {}).get('alignment_id', None)
    if _alignment_id:
        assert alignment_id is None or alignment_id == _alignment_id, (
                'Mismatch between requested ``alignment_id`` (%s) and the one required by evaluation_id %s (%s).' %
                (alignment_id, evaluation_id, _alignment_id)
        )
        alignment_id = _alignment_id
    else:
        alignment_id = get_action_id(cfg, 'align', alignment_id)

    # Parcellate
    parcellation_id = get_action_id(cfg, 'parcellation', parcellation_id)

    return parcellation_id, alignment_id, evaluation_id, aggregation_id


def candidate_name_sort_key(candidate_name):
    trailing_digits = TRAILING_DIGITS.match(candidate_name).group(1)
    if trailing_digits:
        return int(trailing_digits)
    return -1


def join(*args):
    args = [os.path.normpath(x) for x in args]

    return os.path.normpath(os.path.join(*args))


def basename(path):
    path = os.path.normpath(path)

    return os.path.basename(path)


def dirname(path):
    path = os.path.normpath(path)

    return os.path.dirname(path)


def get_suffix(compressed):
    suffix = '.nii'
    if compressed:
        suffix += '.gz'
    return suffix


def get_grid_id(grid_setting):
    grid_id = '_'.join(['%s%s' % (x, grid_setting[x]) for x in grid_setting])

    return grid_id


def get_parcellation_path(output_dir, compressed=True):
    suffix = get_suffix(compressed)
    return join(output_dir, '%s%s' % (SAMPLE_FILENAME_BASE, suffix))


def get_alignment_path(output_dir, alignment_id, compressed=True):
    suffix = get_suffix(compressed)
    alignment_subdir = '%s_%s' % (ALIGNMENT_SUBDIR, alignment_id)
    if basename(output_dir) != alignment_subdir:
        output_dir = join(output_dir, alignment_subdir)

    return join(output_dir, '%s%s' % (ALIGNMENT_FILENAME_BASE, suffix))\


def get_evaluation_path(output_dir, evaluation_id):
    evaluation_subdir = '%s_%s' % (EVALUATION_SUBDIR, evaluation_id)

    return join(output_dir, evaluation_subdir, '%s' % EVALUATION_FILENAME)


def get_aggregation_path(output_dir, aggregation_id):
    aggregation_subdir = '%s_%s' % (AGGREGATION_SUBDIR, aggregation_id)
    if basename(output_dir) != aggregation_subdir:
        output_dir = join(output_dir, aggregation_subdir)

    return join(output_dir, AGGREGATION_FILENAME)


def get_parcellation_mtime(output_dir, compressed=True):
    parcellation_path = get_parcellation_path(output_dir, compressed=compressed)
    if os.path.exists(parcellation_path):
        return os.path.getmtime(parcellation_path)

    return None


def get_alignment_mtime(output_dir, alignment_id, compressed=True):
    alignment_path = get_alignment_path(output_dir, alignment_id, compressed=compressed)
    if os.path.exists(alignment_path):
        return os.path.getmtime(alignment_path)

    return None


def get_evaluation_mtime(output_dir, evaluation_id):
    evaluation_path = get_evaluation_path(output_dir, evaluation_id)
    if os.path.exists(evaluation_path):
        return os.path.getmtime(evaluation_path)

    return None


def get_aggregation_mtime(output_dir, aggregation_id):
    aggregation_path = get_aggregation_path(output_dir, aggregation_id)
    if os.path.exists(aggregation_path):
        return os.path.getmtime(aggregation_path)

    return None


def get_max_mtime(*mtimes):
    mtimes = [x for x in mtimes if x is not None]
    if mtimes:
        return max(mtimes)

    return None


def is_stale(target_mtime, dep_mtime):
    return target_mtime and dep_mtime and target_mtime < dep_mtime


def check_parcellation(output_dir, compressed=True):
    mtime = get_parcellation_mtime(output_dir, compressed=compressed)
    exists = mtime is not None

    return mtime, exists


def check_alignment(output_dir, alignment_id, compressed=True):
    alignment_mtime = get_alignment_mtime(output_dir, alignment_id, compressed=compressed)
    exists = alignment_mtime is not None
    parcellation_mtime = get_parcellation_mtime(output_dir, compressed=compressed)

    if is_stale(alignment_mtime, parcellation_mtime):
        return 1, exists

    mtime = get_max_mtime(alignment_mtime, parcellation_mtime)

    return mtime, exists


def check_evaluation(output_dir, alignment_id, evaluation_id, compressed=True):
    evaluation_mtime = get_evaluation_mtime(output_dir, evaluation_id)
    exists = evaluation_mtime is not None
    alignment_max_mtime, _ = check_alignment(output_dir, alignment_id, compressed=compressed)
    if alignment_max_mtime == 1:
        return 1, exists

    if is_stale(evaluation_mtime, alignment_max_mtime):
        return 1, exists

    mtime = get_max_mtime(evaluation_mtime, alignment_max_mtime)

    return mtime, exists


def check_aggregation(output_dir, alignment_id, evaluation_id, aggregation_id, grid_params, compressed=True):
    aggregation_mtime = get_aggregation_mtime(output_dir, aggregation_id)
    exists = aggregation_mtime is not None
    grid_settings = process_grid_params(grid_params)
    mtime = -np.inf
    for grid_setting in grid_settings:
        grid_id = get_grid_id(grid_setting)
        _output_dir = join(output_dir, GRID_SUBDIR, grid_id)
        evaluation_max_mtime, _ = check_evaluation(
            _output_dir,
            alignment_id,
            evaluation_id,
            compressed=compressed
        )
        if evaluation_max_mtime == 1:
            return 1, exists

        if is_stale(aggregation_mtime, evaluation_max_mtime):
            return 1, exists

        _mtime = get_max_mtime(aggregation_mtime, evaluation_max_mtime)
        if _mtime:
            mtime = max(mtime, _mtime)

    return mtime, exists

