import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def get_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=Loader)

    return cfg

def get_parcellate_kwargs(cfg):
    assert 'parcellate' in cfg, 'get_parcellate_kwargs requires a ``parcellate`` field in cfg.'
    kwargs = {
        'output_dir': cfg.get('output_dir', None),
        'compress_outputs': cfg.get('compress_outputs', True)
    }
    kwargs.update(cfg['parcellate'])

    return kwargs


def get_align_kwargs(cfg, alignment_id):
    assert 'align' in cfg, 'get_align_kwargs requires an ``align`` field in cfg.'
    kwargs = {
        'output_dir': cfg.get('output_dir', None),
        'compress_outputs': cfg.get('compress_outputs', True)
    }
    kwargs.update(cfg['align'][alignment_id])

    return kwargs


def get_evaluate_kwargs(cfg, evaluation_id):
    assert 'evaluate' in cfg, 'get_evaluate_kwargs requires an ``evaluate`` field in cfg.'
    kwargs = {
        'output_dir': cfg.get('output_dir', None),
        'compress_outputs': cfg.get('compress_outputs', True)
    }
    kwargs.update(cfg['evaluate'][evaluation_id])

    return kwargs


def get_aggregate_kwargs(cfg, aggregation_id):
    assert 'aggregate' in cfg, 'get_aggregate_kwargs requires an ``aggregate`` field in cfg.'
    kwargs = {
        'output_dir': cfg.get('output_dir', None),
        'compress_outputs': cfg.get('compress_outputs', True)
    }
    kwargs.update(cfg['aggregate'][aggregation_id])

    return kwargs


def get_refit_kwargs(cfg):
    assert 'refit' in cfg, 'get_refit_kwargs requires a ``refit`` field in cfg.'
    kwargs = cfg['refit']
    if kwargs == True:  # Have to check against True literal because kwargs can also be a dict
        kwargs = {}

    return kwargs


def get_grid_params(cfg):
    if 'grid' in cfg:
        grid_params = cfg['grid']
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
