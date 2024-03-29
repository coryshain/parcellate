import os
import sys
import shutil
import copy
import time
import inspect
import resource
import yaml
import numpy as np
from scipy import signal
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import jaccard_score
from nilearn import image
from fast_poibin import PoiBin

from parcellate.cfg import *
from parcellate.data import *
from parcellate.util import *


######################################
#
#  CORE METHODS
#
######################################


def sample(
        output_dir,
        functional_paths,
        n_networks=50,
        fwhm=None,
        sample_id=None,
        mask_path=None,
        standardize=True,
        normalize=False,
        detrend=False,
        tr=2,
        low_pass=0.1,
        high_pass=0.01,
        n_samples=100,
        clustering_kwargs=None,
        compress_outputs=True,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(sample_id, str), 'sample_id must be given as a str'

    t0 = time.time()

    stderr('%sSampling (sample_id=%s)\n' % (' ' * (indent * 2), sample_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    sample_dir = get_path(output_dir, 'subdir', 'sample', sample_id)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if dump_kwargs:
        kwargs = dict(
            output_dir=output_dir,
            functional_paths=functional_paths,
            n_networks=n_networks,
            fwhm=fwhm,
            sample_id=sample_id,
            mask_path=mask_path,
            standardize=standardize,
            normalize=normalize,
            detrend=detrend,
            tr=tr,
            low_pass=low_pass,
            high_pass=high_pass,
            n_samples=n_samples,
            clustering_kwargs=clustering_kwargs,
            compress_outputs=compress_outputs
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'sample', sample_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'sample', sample_id, compressed=compress_outputs)

    if clustering_kwargs is None:
        clustering_kwargs = dict(
            n_init=N_INIT,
            init_size=INIT_SIZE
        )
        kwargs['clustering_kwargs'] = clustering_kwargs

    input_data = InputData(
        functional_paths=functional_paths,
        fwhm=fwhm,
        mask_path=mask_path,
        standardize=standardize,
        normalize=normalize,
        detrend=detrend,
        tr=tr,
        low_pass=low_pass,
        high_pass=high_pass
    )
    v = input_data.v
    timecourses = input_data.timecourses

    df = pd.DataFrame([dict(n_trs=input_data.n_trs, n_runs=input_data.n_runs)])
    eval_path = get_path(output_dir, 'evaluation', 'sample', sample_id)
    df.to_csv(eval_path, index=False)

    # Sample parcellations by clustering the voxel timecourses
    if n_networks > 256:
        dtype=np.uint16
    else:
        dtype=np.uint8
    samples = np.zeros((v, n_samples), dtype=dtype)  # Shape: <n_samples, n_networks, n_voxels>
    for i in range(n_samples):
        stderr('\r%sSample %d/%d' % (' ' * (indent * 2), i + 1, n_samples))
        m = MiniBatchKMeans(n_clusters=n_networks, **clustering_kwargs)
        _sample = m.fit_predict(timecourses)
        samples[:, i] = _sample
    stderr('\n')
    samples = input_data.unflatten(samples)
    samples.to_filename(output_path)

    stderr('%sSampling time: %ds\n' % (' ' * (indent * 2), time.time() - t0))


def align(
        output_dir,
        reference_atlases,
        alignment_id=None,
        sample_id=None,
        max_subnetworks=None,
        minmax_normalize=True,
        use_poibin=True,
        eps=1e-3,
        compress_outputs=True,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(alignment_id, str), 'alignment_id must be given as a str'
    assert isinstance(sample_id, str), 'sample_id must be given as a str'

    t0 = time.time()

    stderr('%sAligning (alignment_id=%s)\n' % (' ' * (indent * 2), alignment_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    alignment_dir = get_path(output_dir, 'subdir', 'align', alignment_id)
    if not os.path.exists(alignment_dir):
        os.makedirs(alignment_dir)
    if dump_kwargs:
        kwargs = dict(
            reference_atlases=reference_atlases,
            alignment_id=alignment_id,
            sample_id=sample_id,
            max_subnetworks=max_subnetworks,
            minmax_normalize=minmax_normalize,
            use_poibin=use_poibin,
            eps=eps,
            compress_outputs=compress_outputs,
            output_dir=output_dir,
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'align', alignment_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'align', alignment_id, compressed=compress_outputs)
    evaluation_path = get_path(output_dir, 'evaluation', 'align', alignment_id)
    sample_path = get_path(output_dir, 'output', 'sample', sample_id, compressed=compress_outputs)
    assert os.path.exists(sample_path), 'Sample file %s not found' % sample_path

    reference_data = ReferenceData(
        reference_atlases=reference_atlases,
        compress_outputs=compress_outputs
    )
    reference_atlas_names = reference_data.reference_atlas_names
    reference_atlases = reference_data.reference_atlases
    v = reference_data.v
    reference_data.save_atlases(alignment_dir)

    samples = reference_data.flatten(image.smooth_img(sample_path, None))
    n_networks = int(samples.max() + 1)
    if not max_subnetworks:
        max_subnetworks = n_networks
    n_samples = samples.shape[-1]
    samples = samples.T  # Shape: <n_samples, v>, values are integer network indices

    # We do a sparse alignment with slow python loops to avoid OOM for large n_samples or n_networks

    # Rank samples by average best alignment to reference atlas(es)
    sample_scores = np.zeros(n_samples)
    _reference_atlases = np.stack(
        [reference_atlases[x] for x in reference_atlases],
        axis=0
    )
    _reference_atlases_z = standardize_array(_reference_atlases)
    for si in range(n_samples):
        s = samples[si][None, ...] == np.arange(n_networks)[..., None]
        s_z = standardize_array(s)
        scores = np.dot(
            s_z,
            _reference_atlases_z.T
        ) / v
        scores = np.tanh(np.arctanh(scores * (1 - 2 * eps) + eps).mean(axis=-1))
        r = scores.max()
        sample_scores[si] = r
    # Score samples by average best alignment to reference atlas(es)
    sample_scores = minmax_normalize_array(sample_scores)
    # Select reference sample
    ref_ix = np.argmax(sample_scores)
    # Align to reference
    parcellation = align_samples(samples, ref_ix, w=sample_scores)

    # Find candidate network(s) for each reference
    n_reference_atlases = len(reference_atlas_names)
    reference_atlas_scores = np.full((n_reference_atlases,), -np.inf)
    candidates = {}
    results = []
    for j, reference_atlas_name in enumerate(reference_atlas_names):
        reference_atlas = reference_atlases[reference_atlas_name]
        scores = np.zeros(n_networks)
        for ni in range(n_networks):
            scores[ni] = np.corrcoef(parcellation[ni], reference_atlas)[0, 1]
        reference_ix = np.argsort(scores, axis=-1)[::-1]
        candidate = None
        r_prev = -np.inf
        candidate_list = []
        candidate_scores = []
        for ni in range(max_subnetworks):
            ix = reference_ix[..., ni]
            candidate_scores.append(scores[ix])
            _candidate = candidate
            candidate = parcellation[ix]
            candidate = np.clip(candidate, 0, 1)
            candidate_list.append(candidate)
            if use_poibin and ni > 0:
                __candidate = np.zeros(v)
                for _v in range(v):
                    p = 1 - PoiBin([c[_v] for c in candidate_list]).cdf[0]
                    __candidate[_v] = p
                candidate = __candidate
            else:
                candidate = np.clip(
                    np.stack(candidate_list, axis=-1).sum(axis=-1), 0, 1
                )
            r = np.corrcoef(candidate, reference_atlas)[0, 1]
            if r <= r_prev:
                candidate_list = candidate_list[:-1]
                candidate_scores = candidate_scores[:-1]
                candidate = _candidate
                r = r_prev
                break
            r_prev = r

        reference_atlas_scores[j] = r
        candidate_list.insert(0, candidate)
        candidate_scores.insert(0, r)

        for s, candidate in enumerate(candidate_list):
            row = {
                'parcel': reference_atlas_name if s == 0 else '%s_sub%d' % (reference_atlas_name, s),
                'atlas': reference_atlas_name,
                'atlas_score': candidate_scores[s]
            }
            if s == 0:
                row['parcel_type'] = 'network'
            else:
                row['parcel_type'] = 'subnetwork%d' % s
            results.append(row)
            if minmax_normalize:
                candidate = minmax_normalize_array(candidate)
                candidate_list[s] = candidate
            candidate = reference_data.unflatten(candidate)
            if s == 0:
                suffix = ''
            else:
                suffix = '_sub%d' % s
            suffix += get_suffix(compress_outputs)
            candidate.to_filename(join(alignment_dir, '%s%s' % (reference_atlas_name, suffix)))

        candidates[reference_atlas_name] = candidate_list

    results = pd.DataFrame(results)
    results.to_csv(evaluation_path, index=False)

    _parcellation = reference_data.unflatten(parcellation.T)
    _parcellation.to_filename(output_path)

    stderr('%sAlignment time: %ds\n' % (' ' * (indent * 2), time.time() - t0))


def evaluate(
        output_dir,
        evaluation_atlases,
        evaluation_id=None,
        alignment_id=None,
        average_first=False,
        use_poibin=True,
        compress_outputs=True,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(evaluation_id, str), 'evaluation_id must be given as a str'
    assert isinstance(alignment_id, str), 'alignment_id must be given as a str'

    t0 = time.time()

    stderr('%sEvaluating (evaluation_id=%s)\n' % (' ' * (indent * 2), evaluation_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    evaluation_dir = get_path(output_dir, 'subdir', 'evaluate', evaluation_id)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    if dump_kwargs:
        kwargs = dict(
            output_dir=output_dir,
            evaluation_atlases=evaluation_atlases,
            evaluation_id=evaluation_id,
            alignment_id=alignment_id,
            average_first=average_first,
            use_poibin=use_poibin,
            compress_outputs=compress_outputs,
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'evaluate', evaluation_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'evaluate', evaluation_id, compressed=compress_outputs)

    suffix = get_suffix(compress_outputs)

    # Collect references atlases and alignments
    alignment_dir = get_path(output_dir, 'subdir', 'align', alignment_id)
    reference_atlases = []
    candidates = {}
    for reference_atlas in evaluation_atlases:
        reference_atlas_path = join(alignment_dir, '%s%s.nii' % (REFERENCE_ATLAS_PREFIX, reference_atlas))
        if not os.path.exists(reference_atlas_path):
            reference_atlas_path = join(alignment_dir, '%s%s.nii.gz' % (REFERENCE_ATLAS_PREFIX, reference_atlas))
        assert os.path.exists(reference_atlas_path), 'Reference atlas %s not found' % reference_atlas_path
        reference_atlases.append({reference_atlas: reference_atlas_path})

        for path in os.listdir(alignment_dir):
            if path.startswith(reference_atlas):
                if path.endswith(suffix):
                    trim = len(suffix)
                    name = path[:-trim]
                    path = join(alignment_dir, path)
                    if reference_atlas not in candidates:
                        candidates[reference_atlas] = {}
                    candidates[reference_atlas][name] = get_nii(path, add_to_cache=False)

    # Format data
    reference_data = ReferenceData(
        reference_atlases=reference_atlases,
        compress_outputs=compress_outputs
    )
    reference_atlases = reference_data.reference_atlases
    evaluation_data = EvaluationData(
        evaluation_atlases=evaluation_atlases,
        compress_outputs=compress_outputs
    )
    evaluation_atlases = evaluation_data.evaluation_atlases
    evaluation_data.save_atlases(evaluation_dir)
    for x in candidates:
        for y in candidates[x]:
            candidates[x][y] = reference_data.flatten(candidates[x][y])

    stderr(' ' * (indent * 2) + 'Results:\n')
    results = []
    for reference_atlas_name in evaluation_atlases:
        # Score reference atlas as if it were a candidate parcellation (baseline)
        reference_atlas = reference_atlases[reference_atlas_name]
        atlas = reference_atlas
        atlas_name = '%s%s' % (REFERENCE_ATLAS_PREFIX, reference_atlas_name)
        row = _get_evaluation_row(
            atlas,
            atlas_name,
            reference_atlas_name,
            reference_atlas=reference_atlas,
            evaluation_atlases=evaluation_atlases
        )
        row['parcel_type'] = 'baseline'
        results.append(row)
        stderr(_pretty_print_evaluation_row(row, indent=indent + 1) + '\n')

        # Score evaluation atlases as if they were candidate parcellations (baseline)
        for evaluation_atlas_name in evaluation_atlases[reference_atlas_name]:
            atlas = evaluation_atlases[reference_atlas_name][evaluation_atlas_name]
            atlas_name = evaluation_atlas_name
            row = _get_evaluation_row(
                atlas,
                atlas_name,
                reference_atlas_name,
                reference_atlas=reference_atlas,
                evaluation_atlases=evaluation_atlases
            )
            row['parcel_type'] = 'baseline'
            results.append(row)

        # Score candidate parcellations
        candidate_names = sorted(
            list(candidates[reference_atlas_name].keys()),
            key=candidate_name_sort_key
        )
        for c, atlas_name in enumerate(candidate_names):
            atlas = candidates[reference_atlas_name][atlas_name]
            row = _get_evaluation_row(
                atlas,
                atlas_name,
                reference_atlas_name,
                reference_atlas=reference_atlas,
                evaluation_atlases=evaluation_atlases
            )
            if c == 0:
                row['parcel_type'] = 'network'
            else:
                row['parcel_type'] = 'subnetwork%d' % c
            results.append(row)
            if c == 0 or (len(candidate_names) > 2):
                stderr(_pretty_print_evaluation_row(row, indent=indent + 1) + '\n')

    results = pd.DataFrame(results)
    results.to_csv(output_path, index=False)

    stderr('%sEvaluation time: %ds\n' % (' ' * (indent * 2), time.time() - t0))


def aggregate(
        output_dir,
        action_sequence,
        grid_params,
        aggregation_id=None,
        evaluation_id=None,
        alignment_id=None,
        subnetwork_id=None,
        kernel_radius=5,
        eps=1e-3,
        compress_outputs=None,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(output_dir, str), 'output_dir must be given as a str'
    assert isinstance(grid_params, dict), 'grid_params must be given as a dict'

    t0 = time.time()
    stderr('%sAggregating grid\n' % (' ' * (indent * 2)))
    indent += 1

    _alignment_id = get_action_attr('align', action_sequence, 'id')
    if alignment_id is None:
        alignment_id = _alignment_id
    else:
        assert alignment_id == _alignment_id, ('Mismatch between provided alignment_id (%s) '
            'and the one contained in action_sequence (%s).' % alignment_id, _alignment_id)

    _evaluation_id = get_action_attr('evaluate', action_sequence, 'id')
    if evaluation_id is None:
        evaluation_id = _evaluation_id
    else:
        assert evaluation_id == _evaluation_id, ('Mismatch between provided evaluation_id (%s) '
            'and the one contained in action_sequence (%s).' % evaluation_id, _evaluation_id)

    _aggregation_id = get_action_attr('aggregate', action_sequence, 'id')
    if aggregation_id is None:
        aggregation_id = _aggregation_id
    else:

        assert aggregation_id == _aggregation_id, ('Mismatch between provided aggregation_id (%s) '
            'and the one contained in action_sequence (%s).' % aggregation_id, _aggregation_id)

    aggregation_dir = get_path(output_dir, 'subdir', 'aggregate', aggregation_id)
    if not os.path.exists(aggregation_dir):
        os.makedirs(aggregation_dir)
    if dump_kwargs:
        kwargs = dict(
            output_dir=output_dir,
            action_sequence=action_sequence,
            grid_params=grid_params,
            subnetwork_id=subnetwork_id,
            weight_radius=kernel_radius,
            eps=eps,
            compress_outputs=compress_outputs
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'aggregate', aggregation_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'aggregate', aggregation_id, compressed=compress_outputs)
    evaluation_path = get_path(output_dir, 'evaluation', 'aggregate', aggregation_id)

    grid_settings = get_iterator_from_grid_params(grid_params)
    grid_array, ix2val = get_grid_array_from_grid_params(grid_params)
    val2ix = {x: {y: i for i, y in enumerate(ix2val[x])} for x in ix2val}
    grid_keys = sorted(list(ix2val.keys()))
    results = []
    grid_ids = []
    scores = []
    for grid_setting in grid_settings:
        grid_id = get_grid_id(grid_setting)
        grid_ids.append(grid_id)
        _output_dir = get_path(output_dir, 'subdir', 'grid', grid_id)
        score = None
        if evaluation_id is not None:
            evaluation_dir = get_path(_output_dir, 'subdir', 'evaluate', evaluation_id)
            if os.path.exists(evaluation_dir):
                results_file_path = get_path(_output_dir, 'output', 'evaluate', evaluation_id)
                _results = pd.read_csv(results_file_path)
                _results['grid_id'] = grid_id
                results.append(_results)
                score = _get_atlas_score_from_df(_results, subnetwork_id=subnetwork_id, eps=eps)
        else:
            alignment_dir = get_path(_output_dir, 'subdir', 'align', alignment_id)
            if os.path.exists(alignment_dir):
                results_file_path = get_path(_output_dir, 'evaluation', 'align', alignment_id)
                _results = pd.read_csv(results_file_path)
                _results['grid_id'] = grid_id
                results.append(_results)
                score = _get_atlas_score_from_df(_results, subnetwork_id=subnetwork_id, eps=eps)
            else:
                raise ValueError(('No available selection criteria for grid_id %s (no alignment or evaluation '
                                  'data found). Aggregation failed.' % grid_id))
        scores.append(score)
        ix = tuple([val2ix[x][grid_setting[x]] for x in grid_keys])
        grid_array[ix] = score
    results = pd.concat(results, axis=0)

    # Weighted average
    if kernel_radius > 1:
        grid_array = smooth(grid_array, kernel_radius=kernel_radius)

    # Select configuration
    best_ix = np.unravel_index(np.argmax(grid_array), grid_array.shape)
    best_setting = {x: ix2val[x][best_ix[i]] for i, x in enumerate(grid_keys)}
    best_id = get_grid_id(best_setting)
    best_score = grid_array[best_ix]

    results['selected'] = (results.grid_id == best_id) & (results.parcel_type != 'baseline')
    results.to_csv(evaluation_path, index=False)

    # Save configuration
    best_grid_dir = get_path(output_dir, 'subdir', 'grid', best_id)
    _action_sequence = []
    for action in action_sequence:
        action_type, action_id = action['type'], action['id']
        if action_type != 'aggregate':
            action = copy.deepcopy(action)
            kwargs_path = get_path(best_grid_dir, 'kwargs', action_type, action_id)
            exists = os.path.exists(kwargs_path)
            if action_type in ('sample', 'align', 'aggregate'):
                assert exists, '%s does not exist' % kwargs_path
            elif action_type == 'evaluate' and evaluation_id is None:
                assert exists, '%s does not exist' % kwargs_path
            kwargs = get_cfg(kwargs_path)
            kwargs.update(dict(
                output_dir=output_dir
            ))
            action['kwargs'] = kwargs
            _action_sequence.append(action)

    parcellate_kwargs = dict(
        output_dir=output_dir,
        action_sequence=_action_sequence,
        grid_params=None,
        eps=eps,
    )

    with open(output_path, 'w') as f:
        yaml.safe_dump(parcellate_kwargs, f, sort_keys=False)

    stderr('%sBest grid_id: %s | atlas score: %0.3f\n' % (' ' * (indent * 2), best_id, best_score))

    stderr('%sAggregation time: %ds\n' % (' ' * (indent * 2), time.time() - t0))


def parcellate(
        output_dir,
        action_sequence,
        grid_params=None,
        eps=1e-3,
        compress_outputs=True,
        overwrite=False,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(output_dir, str), 'output_dir is required, must be given as a str'
    assert isinstance(action_sequence, list), ('action_sequence is required, must be given as a list of dict'
        'and grid_params must be provided as dicts, or neither can be.')

    validate_action_sequence(action_sequence)

    t0 = time.time()
    stderr('%sParcellating\n' % (' ' * (indent * 2)))
    indent += 1

    sample_id = get_action_attr('sample', action_sequence, 'id')
    alignment_id = get_action_attr('align', action_sequence, 'id')
    evaluation_id = get_action_attr('evaluate', action_sequence, 'id')
    aggregation_id = get_action_attr('aggregate', action_sequence, 'id')
    parcellation_id = get_action_attr('parcellate', action_sequence, 'id')

    assert isinstance(sample_id, str), 'sample_id is required, must be given as a str.'
    assert isinstance(alignment_id, str), 'alignment_id is required, must be given as a str.'
    assert isinstance(parcellation_id, str), 'parcellation_id is required, must be given as a str.'

    use_grid = get_action_attr('parcellate', action_sequence, 'kwargs').get('use_grid', True)

    parcellation_dir = get_path(output_dir, 'subdir', 'parcellate', parcellation_id)
    if not os.path.exists(parcellation_dir):
        os.makedirs(parcellation_dir)
    if dump_kwargs:
        kwargs = dict(
            output_dir=output_dir,
            action_sequence=action_sequence,
            grid_params=grid_params,
            eps=eps,
            compress_outputs=compress_outputs,
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'parcellate', parcellation_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'parcellate', parcellation_id, compressed=compress_outputs)

    for action in action_sequence:
        action['kwargs']['output_dir'] = output_dir
        action['kwargs']['compress_outputs'] = compress_outputs

    suffix = get_suffix(compress_outputs)

    # Grid search
    if use_grid and grid_params:
        stderr('%sGrid searching\n' % (' ' * (indent * 2)))

        # Core loop
        indent += 1
        grid_settings = get_iterator_from_grid_params(grid_params)
        for grid_setting in grid_settings:
            grid_id = get_grid_id(grid_setting)

            stderr('%sGrid id: %s\n' % (' ' * (indent * 2), grid_id))

            # Update kwargs
            _output_dir = get_path(output_dir, 'subdir', 'grid', grid_id)
            _action_sequence = []
            for action in action_sequence:
                if action['type'] != 'aggregate':
                    action = copy.deepcopy(action)
                    _kwargs = action['kwargs']
                    _kwarg_keys = set(inspect.signature(ACTIONS[action['type']]).parameters.keys())
                    _grid_setting = {x: grid_setting[x] for x in grid_setting if x in _kwarg_keys}
                    _kwargs.update(_grid_setting)
                    _action_sequence.append(action)

            # Recursion bottoms out since grid_params is None
            parcellate(
                _output_dir,
                _action_sequence,
                eps=eps,
                compress_outputs=compress_outputs,
                overwrite=overwrite,
                indent=indent + 1
            )

        indent -= 1

        # Aggregate
        action = None
        for a, action in enumerate(action_sequence):
            if action['type'] == 'aggregate':
                break
        assert action is not None and action['type'] == 'aggregate', ('action type "aggregate" not found in '
            'action_sequence')
        _action_sequence = []
        for action in action_sequence:
            _action_sequence.append(action)
            if action['type'] == 'aggregate':
                break
        mtime, exists = check_deps(
            output_dir,
            _action_sequence,
            compressed=True
        )
        stale = mtime == 1
        if overwrite or stale or not exists:
            # Recursion bottoms out since grid_params is None
            _aggregate_kwargs = copy.deepcopy(action['kwargs'])
            # _action_sequence.append(get_action('aggregate', action_sequence))
            _aggregate_kwargs.update(dict(
                output_dir=output_dir,
                action_sequence=action_sequence,
                grid_params=grid_params,
                eps=eps,
                compress_outputs=compress_outputs,
                indent=indent,
            ))
            aggregate(**_aggregate_kwargs)
        else:
            stderr('%sAggregation exists. Skipping. To re-aggregate, run with overwrite=True.\n' %
                  (' ' * (indent * 2)))
        aggregation_output_path = get_path(output_dir, 'output', 'aggregate', aggregation_id)
        with open(aggregation_output_path, 'r') as f:
            parcellate_kwargs = yaml.safe_load(f)

        # Parcellate
        kwargs_update = action['kwargs']
        for action in parcellate_kwargs['action_sequence']:
            if action['type'] != 'parcellate':
                _kwarg_keys = set(inspect.signature(ACTIONS[action['type']]).parameters.keys())
                _kwargs_update = {x: kwargs_update[x] for x in kwargs_update if x in _kwarg_keys}
                action['kwargs'].update(_kwargs_update)
        action_prefix = []
        for action in action_sequence[:-1]:  # Add dependencies to grid, ignoring last ('parcellate') action
            action_prefix.append(dict(
                type=action['type'],
                id=action['id'],
                kwargs={}
            ))
        parcellate_kwargs['action_sequence'] = action_prefix + parcellate_kwargs['action_sequence']
        parcellate_kwargs['dump_kwargs'] = False  # Don't let recursive call overwrite top-level kwargs file
        parcellate(**parcellate_kwargs)
    else:
        action_sequence_full = action_sequence
        _action_sequence = []
        for a in range(len(action_sequence) - 1, -1, -1):
            action = action_sequence[a]
            _action_sequence.insert(0, action)
            if action['type'] == 'sample':
                break
        action_sequence = _action_sequence

        sample_id = get_action_attr('sample', action_sequence, 'id')
        alignment_id = get_action_attr('align', action_sequence, 'id')
        evaluation_id = get_action_attr('evaluate', action_sequence, 'id')
        aggregation_id = get_action_attr('aggregate', action_sequence_full, 'id')

        n = len(action_sequence)
        N = len(action_sequence_full)
        for a, action in enumerate(action_sequence):
            action_type = action['type']
            action_kwargs = action['kwargs']

            e = N - n + a + 1

            mtime, exists = check_deps(
                output_dir,
                action_sequence_full[:e],
                compressed=True
            )
            stale = mtime == 1
            if overwrite or stale or not exists:
                do_action = True
            else:
                do_action = False

            if action_type == 'sample':
                if do_action:
                    action_kwargs.update(dict(
                        output_dir=output_dir,
                        sample_id=sample_id
                    ))
                    sample(**action_kwargs, indent=indent)
                else:
                    stderr('%sSample exists. Skipping. To resample, run with overwrite=True.\n' %
                          (' ' * (indent * 2)))
            elif action_type == 'align':
                if do_action:
                    action_kwargs.update(dict(
                        output_dir=output_dir,
                        alignment_id=alignment_id,
                        sample_id=sample_id
                    ))
                    align(**action_kwargs, indent=indent)
                else:
                    stderr('%sAlignment exists. Skipping. To re-align, run with overwrite=True.\n' %
                          (' ' * (indent * 2)))
            elif action_type == 'evaluate':
                if do_action:
                    action_kwargs.update(dict(
                        output_dir=output_dir,
                        evaluation_id=evaluation_id,
                        alignment_id=alignment_id
                    ))
                    evaluate(**action_kwargs, indent=indent)
                else:
                    stderr('%sEvaluation exists. Skipping. To re-evaluate, run with overwrite=True.\n' %
                          (' ' * (indent * 2)))
            elif action_type == 'parcellate':
                # Copy final files to destination
                if do_action:
                    results_copied = False
                    if aggregation_id is not None:
                        parcellation_kwargs_path = get_path(output_dir, 'output', 'aggregate', aggregation_id)
                        assert os.path.exists(parcellation_kwargs_path), ('Aggregation output %s not found' %
                            parcellation_kwargs_path)
                        shutil.copy(parcellation_kwargs_path, join(parcellation_dir, 'parcellate_kwargs_final.yml'))

                    if evaluation_id is not None:
                        evaluation_dir = get_path(output_dir, 'subdir', 'evaluate', evaluation_id)
                        for filename in os.listdir(evaluation_dir):
                            if filename.endswith(suffix) or filename == PATHS['evaluate']['output']:
                                shutil.copy(join(evaluation_dir, filename), join(parcellation_dir, filename))
                                if filename == PATHS['evaluate']['output']:
                                    results_copied = True
                    alignment_dir = get_path(output_dir, 'subdir', 'align', alignment_id)
                    for filename in os.listdir(alignment_dir):
                        if filename.endswith(suffix) or (not results_copied and filename == PATHS['align']['evaluation']):
                            shutil.copy(join(alignment_dir, filename), join(parcellation_dir, filename))
                else:
                    stderr('%sParcellation exists. Skipping. To re-parcellate, run with overwrite=True.\n' %
                          (' ' * (indent * 2)))
            else:
                raise ValueError('Unrecognized action_type %s' % action_type)


        assert os.path.exists(output_path)

    stderr('%sTotal time elapsed: %ds\n' % (' ' * (indent * 2), time.time() - t0))










######################################
#
#  PRIVATE HELPER METHODS
#
######################################


def _get_atlas_score(
        atlas,
        reference_atlas,
):
    r = np.corrcoef(reference_atlas, atlas)[0, 1]

    m, M = atlas.min(), atlas.max()
    if m < 0 or M > 1:
        atlas = minmax_normalize_array(atlas)
    n_voxels = atlas.sum()

    row = dict(
        atlas_score=r,
        n_voxels=n_voxels
    )

    for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        ji = jaccard_score(reference_atlas > p, atlas > p, zero_division=0)
        row['jaccard_atpgt%s' % p] = ji

    return row


def _get_evaluation_spcorr(
        atlas,
        reference_atlas_name,
        evaluation_atlases
):
    row = {}
    _evaluation_atlases = evaluation_atlases[reference_atlas_name]
    evaluation_atlas_names = list(_evaluation_atlases.keys())
    for evaluation_atlas_name in evaluation_atlas_names:
        evaluation_atlas = _evaluation_atlases[evaluation_atlas_name]
        r = np.corrcoef(atlas, evaluation_atlas)[0, 1]
        row['%s_score' % evaluation_atlas_name] = r

    return row


def _get_evaluation_contrasts(
        atlas,
        reference_atlas_name,
        evaluation_atlases
):
    m, M = atlas.min(), atlas.max()
    if m < 0 or M > 1:
        atlas = minmax_normalize_array(atlas)
    row = {}
    _evaluation_atlases = evaluation_atlases[reference_atlas_name]
    evaluation_atlas_names = list(_evaluation_atlases.keys())
    for evaluation_atlas_name in evaluation_atlas_names:
        evaluation_atlas = _evaluation_atlases[evaluation_atlas_name]
        for p in (None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            if p is None:
                _atlas = atlas
                suffix = ''
            else:
                _atlas = atlas > p
                suffix = '_atpgt%s' % p
            denom = _atlas.sum()
            if denom:
                contrast = (_atlas * evaluation_atlas).sum() / denom
            else:
                contrast = 0
            row['%s_contrast%s' % (evaluation_atlas_name, suffix)] = contrast

    return row


def _get_evaluation_row(
        atlas,
        atlas_name,
        reference_atlas_name,
        reference_atlas=None,
        evaluation_atlases=None
):
    row = dict(
        parcel=atlas_name,
        atlas=reference_atlas_name
    )
    if reference_atlas is not None:
        row.update(_get_atlas_score(
            atlas,
            reference_atlas
        ))
    if evaluation_atlases is not None:
        row.update(_get_evaluation_spcorr(
            atlas,
            reference_atlas_name,
            evaluation_atlases
        ))
        row.update(_get_evaluation_contrasts(
            atlas,
            reference_atlas_name,
            evaluation_atlases
        ))

    return row


def _pretty_print_evaluation_row(
        row,
        max_evals=1,
        indent=0
):
    to_print = []
    scores = set()
    contrasts = set()
    for col in row:
        if col == 'parcel':
            to_print.append(row[col])
        elif col == 'n_voxels':
            to_print.append('n voxels: %d' % row[col])
        elif col.endswith('_score'):
            _col = '_'.join(col.split('_')[:-1])
            if max_evals is None or len(scores) < max_evals:
                if col != 'atlas_score':
                    scores.add(col)
                to_print.append('%s score: %0.3f' % (_col, row[col]))
        elif col.endswith('contrast'):
            _col = '_'.join(col.split('_')[:-1])
            if max_evals is None or len(contrasts) < max_evals:
                contrasts.add(col)
                to_print.append('%s contrast: %0.3f' % (_col, row[col]))

    to_print = ' | '.join(to_print)
    to_print = ' ' * (indent * 2) + to_print

    return to_print


def _get_atlas_score_from_df(df_scores, subnetwork_id=None, eps=1e-3):
    reference_atlas_names = df_scores.atlas.unique().tolist()
    parcel_names = df_scores.parcel
    scores = df_scores.atlas_score
    target_parcels = reference_atlas_names
    if subnetwork_id:
        target_parcels = [x + '_sub%d' % subnetwork_id for x in target_parcels]
    sel = parcel_names.isin(target_parcels)
    parcel_names = parcel_names[sel].tolist()

    assert set(target_parcels) == set(parcel_names), (f'Parcel names and reference atlas names mismatch. '
         f'Parcel names: {parcel_names}. Reference atlas names: {reference_atlas_names}')

    scores = scores[sel]
    score = np.tanh(np.arctanh(scores * (1 - 2 * eps) + eps).mean())

    return score








######################################
#
#  CONSTANTS
#
######################################

ACTIONS = dict(
    sample=sample,
    align=align,
    evaluate=evaluate,
    aggregate=aggregate,
    parcellate=parcellate,
)