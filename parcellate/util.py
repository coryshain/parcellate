import sys
import os
import itertools
import numpy as np

from parcellate.constants import *


def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def get_iterator_from_grid_params(grid_params):
    _grid_params = {}
    if grid_params:
        for grid_param in grid_params:
            _grid_params[grid_param] = get_grid_param_value_list(grid_params[grid_param])
    else:
        _grid_params = {'model': [0]}
    keys = sorted(list(_grid_params.keys()))
    cross = itertools.product(*[_grid_params[x] for x in keys])

    for x in cross:
        row = {}
        for _x, key in zip(x, keys):
            row[key] = _x

        yield row

def get_grid_array_from_grid_params(grid_params):
    vals = {}
    if grid_params:
        for grid_param in grid_params:
            vals[grid_param] = get_grid_param_value_list(grid_params[grid_param])

    arr = np.full(tuple([len(vals[x]) for x in sorted(list(vals.keys()))]), np.nan)

    return arr, vals


def get_grid_param_value_list(vals):
    _vals = []
    for val in vals:
        if isinstance(val, list):
            _val = []
            for v in val:
                vi = int(v)
                vf = float(v)
                if vi == vf:  # Integer
                    _val.append(vi)
                else:
                    _val.append(vf)
            __vals = np.arange(*_val).tolist()
        else:
            __vals = [val]
        _vals.extend(__vals)
    return _vals


def get_grid_id(grid_setting):
    grid_id = '_'.join(['%s%s' % (x, grid_setting[x]) for x in grid_setting])

    return grid_id


def smooth(arr, kernel_radius=3):
    out = np.zeros_like(arr)
    if kernel_radius > 1:
        indices = [list(range(x)) for x in arr.shape]
        for index in itertools.product(*indices):
            sel = tuple([slice(max(0, i - kernel_radius + 1), i + kernel_radius) for i in index])
            out[index] = arr[sel].mean()
    return out


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


def validate_action_sequence(action_sequence):
    n = len(action_sequence)
    for a, action in enumerate(action_sequence):
        action_type = action['type']
        action_id = action['id']
        if a == 0:
            assert action_type == 'sample', ('The first action in the sequence must be sample, got %s.' %
                action_type)
        if a < n - 1:  # Not at final action
            assert not action_type == 'parcellate', 'parcellate can only appear at the end of action_sequence'
            next_action = action_sequence[a + 1]
            next_action_type = next_action['type']

            if action_type == 'sample':
                assert next_action_type in ('align', 'label'), ('Got invalid action sequence %s -> %s' %
                    (action_type, next_action_type))
                if 'sample_id' in next_action['kwargs']:
                    assert next_action['kwargs']['sample_id'] == action_id, ('Got sample_id %s, but align expects '
                        'sample_id %s.' % (action_id, next_action['kwargs']['sample_id']))
            elif action_type == 'align':
                assert next_action_type in ('label', 'aggregate', 'parcellate'), ('Got invalid action sequence '
                    '%s -> %s' % (action_type, next_action_type))
                if 'alignment_id' in next_action['kwargs']:
                    assert next_action['kwargs']['alignment_id'] == action_id, ('Got alignment_id %s, but %s expects '
                        'alignment_id %s.' % (action_id, next_action_type, next_action['kwargs']['alignment_id']))
            elif action_type == 'label':
                assert next_action_type in ('evaluate', 'aggregate', 'parcellate'), ('Got invalid action sequence '
                    '%s -> %s' % (action_type, next_action_type))
                if 'labeling_id' in next_action['kwargs']:
                    assert next_action['kwargs']['labeling_id'] == action_id, ('Got labeling_id %s, but %s expects '
                        'labeling_id %s.' % (action_id, next_action_type, next_action['kwargs']['labeling_id']))
            elif action_type == 'evaluate':
                assert next_action_type in ('aggregate', 'parcellate'), ('Got invalid action sequence '
                    '%s -> %s' % (action_type, next_action_type))
                if 'evaluation_id' in next_action['kwargs']:
                    assert next_action['kwargs']['evaluation_id'] == action_id, ('Got evaluation_id %s, but %s expects '
                        'evaluation_id %s.' % (action_id, next_action_type, next_action['kwargs']['evaluation_id']))
            elif action_type == 'aggregate':
                assert next_action_type in ('parcellate', 'sample'), ('Got invalid action sequence %s -> %s' %
                    (action_type, next_action_type))
                if 'aggregation_id' in next_action['kwargs']:
                    assert next_action['kwargs']['aggregation_id'] == action_id, ('Got aggregation_id %s, but %s '
                        'expects aggregation_id %s.' % (action_id, next_action_type,
                        next_action['kwargs']['aggregation_id']))


def get_action(action_type, action_sequence):
    for action in action_sequence:
        if action['type'] == action_type:
            return action
    return None


def get_action_attr(action_type, action_sequence, action_attr):
    action = get_action(action_type, action_sequence)
    if action is None:
        return None
    return action[action_attr]


def is_stale(target_mtime, dep_mtime):
    return target_mtime and dep_mtime and target_mtime < dep_mtime


def check_deps(
        output_dir,
        action_sequence,
        compressed=True
):
    if not len(action_sequence):
        return None, False
    action = action_sequence[-1]  # Iterate from end
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
                action_sequence[:-1],
                compressed=compressed
            )
            if _dep_mtime == 1:
                return 1, exists
            if is_stale(mtime, _dep_mtime):
                return 1, exists
            dep_mtime = get_max_mtime(dep_mtime, _dep_mtime)
    else:
        dep_mtime, _ = check_deps(output_dir, action_sequence[:-1], compressed=True)
        if dep_mtime == 1:
            return 1, exists
        if is_stale(mtime, dep_mtime):
            return 1, exists

    mtime = get_max_mtime(mtime, dep_mtime)

    return mtime, exists


def get_overwrite(overwrite):
    if isinstance(overwrite, dict):
        return overwrite

    out = dict(
        sample=False,
        align=False,
        label=False,
        evaluate=False,
        aggregate=False,
        parcellate=False
    )
    if overwrite is None:
        for x in out:
            out[x] = True
    elif isinstance(overwrite, str):
        out[overwrite] = True
    elif overwrite is False:
        pass
    else:
        raise ValueError('Unrecognized value for overwrite: %s' % overwrite)

    return out
