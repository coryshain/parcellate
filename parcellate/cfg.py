import yaml
import copy

from parcellate.constants import ACTION_VERB_TO_NOUN


def get_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

def get_kwargs(cfg, action_type, action_id):
    if action_id is not None and action_type in cfg:
        kwargs = copy.deepcopy(cfg[action_type][action_id])
        kwargs.update({'%s_id' % ACTION_VERB_TO_NOUN[action_type]: action_id})
    else:
        kwargs = None

    return kwargs


def get_grid_params(cfg):
    if 'grid' in cfg:
        grid_params = copy.deepcopy(cfg['grid'])
    else:
        grid_params = None

    assert not 'id' in grid_params, 'grid_params contained key "id", which is a reserved keyword'

    return grid_params


def get_val_from_kwargs(
        key,
        parcellate_kwargs=None,
        align_kwargs=None,
        evaluate_kwargs=None,
        aggregate_kwargs=None
):
    val = None
    kwargs = dict(
        parcellate_kwargs=parcellate_kwargs,
        align_kwargs=align_kwargs,
        evaluate_kwargs=evaluate_kwargs,
        aggregate_kwargs=aggregate_kwargs
    )
    actions = set()
    for kwarg_type in kwargs:
        if kwargs[kwarg_type]:
            actions.add(kwarg_type)
            val = kwargs[kwarg_type].get(key, None)
            if val is not None:
                break

    return val


def get_action_sequence(
        cfg,
        action_type,
        action_id,
        deps=None
):
    if deps is None:
        deps = []
    if action_id is None:
        action_id = list(cfg[action_type].keys())[0]
    assert action_id in cfg[action_type], 'No entry %s found in %s' % (action_id, action_type)
    dep = {'type': action_type, 'id': action_id}
    deps.append(dep)

    if action_type == 'sample':
        return deps
    if action_type == 'align':
        action_id = cfg[action_type][action_id].get('sample_id', None)
        action_type = 'sample'
    elif action_type == 'evaluate':
        action_id = cfg[action_type][action_id].get('alignment_id', None)
        action_type = 'align'
    elif action_type == 'aggregate':
        if 'evaluation_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('evaluation_id', None)
            action_type = 'evaluate'
        elif 'alignment_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('alignment_id', None)
            action_type = 'align'
        elif 'evaluate' in cfg:
            action_type = 'evaluate'
            action_id = None
        else:
            action_type = 'align'
            action_id = None
    elif action_type == 'parcellate':
        if 'aggregation_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('aggregation_id', None)
            action_type = 'aggregate'
        elif 'evaluation_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('evaluation_id', None)
            action_type = 'evaluate'
        elif 'alignment_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('alignment_id', None)
            action_type = 'align'
        elif 'aggregate' in cfg:
            action_type = 'aggregate'
            action_id = None
        elif 'evaluate' in cfg:
            action_type = 'evaluate'
            action_id = None
        else:
            action_type = 'align'
            action_id = None
    else:
        raise ValueError('Unrecognized action_type %s' % action_type)

    deps = get_action_sequence(
        cfg,
        action_type,
        action_id,
        deps
    )

    return deps
