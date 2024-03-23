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


def get_path(output_dir, path_type, action_type, action_id, compressed=True):
    assert action_type in PATHS, 'Unrecognized action_type %s' % action_type
    assert path_type in PATHS[action_type], 'Unrecognized path_type %s for action_type %s' % (path_type, action_type)
    path = PATHS[action_type][path_type]
    if path.endswith('%s'):
        path = path % get_suffix(compressed)

    if path_type == 'subdir':
        suffix = join(path, action_id)
    else:
        prefix = PATHS[action_type]['subdir']
        suffix = join(prefix, action_id, path)

    path = join(output_dir, suffix)

    return path


def get_mtime(output_dir, action_type, action_id, compressed=True):
    path = get_path(
        output_dir,
        'output',
        action_type,
        action_id,
        compressed=compressed,
    )

    if os.path.exists(path):
        return os.path.getmtime(path)

    return None


def get_max_mtime(*mtimes):
    mtimes = [x for x in mtimes if x is not None]
    if mtimes:
        return max(mtimes)

    return None


def get_action_id(action_type, action_sequence):
    action_id = None
    for dep in action_sequence:
        if dep['type'] == action_type:
            return dep['id']
    return action_id


def is_stale(target_mtime, dep_mtime):
    return target_mtime and dep_mtime and target_mtime < dep_mtime


def check_deps(
        output_dir,
        action_sequence,
        compressed=True
):
    if not len(action_sequence):
        return -1, False
    action = action_sequence[0]
    action_type, action_id = action['type'], action['id']
    mtime = get_mtime(output_dir, action_type, action_id, compressed=compressed)
    exists = mtime is not None
    if action_type == 'aggregate':
        grid_dir = get_path(output_dir, 'subdir', 'grid', '')
        dep_mtime = None
        for grid_id in os.listdir(grid_dir):
            _output_dir = join(grid_dir, grid_id)
            _dep_mtime, _ = check_deps(
                _output_dir,
                action_sequence[1:],
                compressed=compressed
            )
            if _dep_mtime == 1:
                return 1, exists
            if is_stale(mtime, _dep_mtime):
                return 1, exists
            dep_mtime = get_max_mtime(dep_mtime, _dep_mtime)
    else:
        dep_mtime, _ = check_deps(output_dir, action_sequence[1:], compressed=True)
        if dep_mtime == 1:
            return 1, exists
        if is_stale(mtime, dep_mtime):
            return 1, exists

    mtime = get_max_mtime(mtime, dep_mtime)

    return mtime, exists
