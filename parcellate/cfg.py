import yaml
import copy

from parcellate.constants import ACTION_VERB_TO_NOUN


def get_cfg(path):
    """
    Load a YAML configuration file

    :param path: ``str``; path to the YAML configuration file
    :return: ``dict``; configuration dictionary
    """

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_kwargs(cfg, action_type, action_id):
    """
    Get the function kwargs for a given action type and action ID

    :param cfg: ``dict``; configuration dictionary
    :param action_type: ``str``; action type
    :param action_id: ``str``; action ID
    :return: ``dict``; function kwargs
    """

    if action_id is not None and action_type in cfg:
        kwargs = copy.deepcopy(cfg[action_type][action_id])
        kwargs.update({'%s_id' % ACTION_VERB_TO_NOUN[action_type]: action_id})
        if 'xfm_path' in cfg and action_type in ('sample', 'label', 'evaluate') and 'xfm_path' not in kwargs:
            kwargs['xfm_path'] = cfg['xfm_path']
        if 'mask_path' in cfg and action_type in ('sample', 'align', 'label', 'evaluate') and 'mask_path' not in kwargs:
            kwargs['mask_path'] = cfg['mask_path']
    else:
        kwargs = None

    return kwargs


def get_grid_params(cfg):
    """
    Get the grid search parameters from the configuration dictionary

    :param cfg: ``dict``; configuration dictionary
    :return: ``dict``; grid search parameters
    """

    if 'grid' in cfg:
        grid_params = copy.deepcopy(cfg['grid'])
    else:
        grid_params = None

    return grid_params


def get_val_from_kwargs(
        key,
        parcellate_kwargs=None,
        align_kwargs=None,
        evaluate_kwargs=None,
        aggregate_kwargs=None
):
    """
    Get a value from the kwargs of one of the action types

    :param key: ``str``; key to search for
    :param parcellate_kwargs: ``dict``; parcellate kwargs
    :param align_kwargs: ``dict``; align kwargs
    :param evaluate_kwargs: ``dict``; evaluate kwargs
    :param aggregate_kwargs: ``dict``; aggregate kwargs
    :return: ``object``; value
    """

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
        action_sequence=None
):
    """
    Recursively get the action sequence for a given action type and action ID

    :param cfg: ``dict``; configuration dictionary
    :param action_type: ``str``; action type
    :param action_id: ``str``; action ID
    :param action_sequence: ``list`` of ``dict``; initial value for action sequence
    :return: ``list`` of ``dict``; action sequence
    """

    if action_sequence is None:
        action_sequence = []
    if action_id is None:
        if action_type in cfg:
            action_id = list(cfg[action_type].keys())[0]
        else:
            action_id = 'main'
            cfg[action_type] = {action_id: {}}  # Add empty entry for main action
    assert action_id in cfg[action_type], 'No entry %s found in %s' % (action_id, action_type)
    kwargs = get_kwargs(cfg, action_type, action_id)
    action = dict(
        type=action_type,
        id=action_id,
        kwargs=kwargs
    )
    if len(action_sequence):
        action_sequence[0]['kwargs']['%s_id' % ACTION_VERB_TO_NOUN[action_type]] = action_id
    action_sequence.insert(0, action)

    if action_type == 'sample':
        return action_sequence
    if action_type == 'align':
        action_id = cfg[action_type][action_id].get('sample_id', None)
        action_type = 'sample'
    elif action_type == 'label':
        average_first = cfg[action_type][action_id].get('average_first', True)
        if average_first:
            if 'alignment_id' in cfg[action_type][action_id]:
                action_id = cfg[action_type][action_id].get('alignment_id', None)
                action_type = 'align'
            else:
                action_type = 'align'
                action_id = None
                if not 'align' in cfg:
                    cfg['align'] = {'main': {}}
        else:  # Must be preceded by sample step
            if 'sample_id' in cfg[action_type][action_id]:
                action_id = cfg[action_type][action_id].get('sample_id', None)
                action_type = 'sample'
            else:
                action_type = 'sample'
                action_id = None
                if not 'sample' in cfg:
                    cfg['sample'] = {'main': {}}
    elif action_type == 'evaluate':
        action_id = cfg[action_type][action_id].get('labeling_id', None)
        action_type = 'label'
    elif action_type == 'aggregate':
        if 'evaluation_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('evaluation_id', None)
            action_type = 'evaluate'
        elif 'labeling_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('labeling_id', None)
            action_type = 'label'
        elif 'evaluate' in cfg:
            action_type = 'evaluate'
            action_id = None
        else:
            action_type = 'label'
            action_id = None
    elif action_type == 'parcellate':
        if ('aggregation_id' in cfg[action_type][action_id] and
                cfg[action_type][action_id]['aggregation_id'] is not None):
            action_id = cfg[action_type][action_id].get('aggregation_id', None)
            action_type = 'aggregate'
        elif 'evaluation_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('evaluation_id', None)
            action_type = 'evaluate'
        elif 'labeling_id' in cfg[action_type][action_id]:
            action_id = cfg[action_type][action_id].get('labeling_id', None)
            action_type = 'label'
        elif ('aggregate' in cfg and 'grid' in cfg and len(cfg['grid']) and
                cfg[action_type][action_id].get('aggregation_id', 'not found') is not None):
            action_type = 'aggregate'
            action_id = None
        elif 'evaluate' in cfg:
            action_type = 'evaluate'
            action_id = None
        else:
            action_type = 'label'
            action_id = None
    else:
        raise ValueError('Unrecognized action_type %s' % action_type)

    action_sequence = get_action_sequence(
        cfg,
        action_type,
        action_id,
        action_sequence
    )

    return action_sequence
