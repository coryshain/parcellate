import os
import sys
import shutil
import copy
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
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
        n_networks,
        functional_paths,
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
        indent=0,
        **kwargs  # Ignored
):
    assert isinstance(sample_id, str), 'sample_id must be given as a str'

    t0 = time.time()

    print('%sSampling (sample_id=%s)' % (' ' * (indent * 2), sample_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    kwargs = dict(
        output_dir=output_dir,
        n_networks=n_networks,
        functional_paths=functional_paths,
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

    sample_dir = get_path(output_dir, 'subdir', 'sample', sample_id)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
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

    # Sample parcellations by clustering the voxel timecourses
    if n_networks > 256:
        dtype=np.uint16
    else:
        dtype=np.uint8
    samples = np.zeros((v, n_samples), dtype=dtype)  # Shape: <n_samples, n_networks, n_voxels>
    for i in range(n_samples):
        sys.stdout.write('\r%sSample %d/%d' % (' ' * (indent * 2), i + 1, n_samples))
        sys.stdout.flush()
        m = MiniBatchKMeans(n_clusters=n_networks, **clustering_kwargs)
        _sample = m.fit_predict(timecourses)
        samples[:, i] = _sample
    sys.stdout.write('\n')
    sys.stdout.flush()
    samples = input_data.unflatten(samples)
    samples.to_filename(output_path)

    print('%sSampling time: %ds' % (' ' * (indent * 2), time.time() - t0))

    return samples


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
        indent=0,
        **kwargs  # Ignored
):
    assert isinstance(alignment_id, str), 'alignment_id must be given as a str'
    assert isinstance(sample_id, str), 'sample_id must be given as a str'

    t0 = time.time()

    print('%sAligning (alignment_id=%s)' % (' ' * (indent * 2), alignment_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

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

    alignment_dir = get_path(output_dir, 'subdir', 'align', alignment_id)
    if not os.path.exists(alignment_dir):
        os.makedirs(alignment_dir)
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

    print('%sAlignment time: %ds' % (' ' * (indent * 2), time.time() - t0))

    return parcellation


def evaluate(
        output_dir,
        evaluation_atlases,
        evaluation_id=None,
        alignment_id=None,
        average_first=False,
        use_poibin=True,
        compress_outputs=True,
        indent=0,
        **kwargs  # Ignored
):
    assert isinstance(evaluation_id, str), 'evaluation_id must be given as a str'
    assert isinstance(alignment_id, str), 'alignment_id must be given as a str'

    t0 = time.time()

    print('%sEvaluating (evaluation_id=%s)' % (' ' * (indent * 2), evaluation_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    kwargs = dict(
        output_dir=output_dir,
        evaluation_atlases=evaluation_atlases,
        evaluation_id=evaluation_id,
        alignment_id=alignment_id,
        average_first=average_first,
        use_poibin=use_poibin,
        compress_outputs=compress_outputs,
    )

    evaluation_dir = get_path(output_dir, 'subdir', 'evaluate', evaluation_id)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
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
        reference_atlas_path = join(alignment_dir, 'reference_atlas_%s.nii' % reference_atlas)
        if not os.path.exists(reference_atlas_path):
            reference_atlas_path = join(alignment_dir, 'reference_atlas_%s.nii.gz' % reference_atlas)
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
                    candidates[reference_atlas][name] = get_nii(path)

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

    print(' ' * (indent * 2) + 'Results:')
    results = []
    for reference_atlas_name in evaluation_atlases:
        # Score reference atlas as if it were a candidate parcellation (baseline)
        reference_atlas = reference_atlases[reference_atlas_name]
        atlas = reference_atlas
        atlas_name = 'reference_atlas_%s' % reference_atlas_name
        row = _get_evaluation_row(
            atlas,
            atlas_name,
            reference_atlas_name,
            reference_atlas=reference_atlas,
            evaluation_atlases=evaluation_atlases
        )
        row['parcel_type'] = 'baseline'
        results.append(row)
        print(_pretty_print_evaluation_row(row, indent=indent + 1))

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
                print(_pretty_print_evaluation_row(row, indent=indent + 1))

    results = pd.DataFrame(results)
    results.to_csv(output_path, index=False)

    print('%sEvaluation time: %ds' % (' ' * (indent * 2), time.time() - t0))

    return results


def aggregate(
        output_dir,
        grid_params,
        sample_id,
        alignment_id,
        aggregation_id,
        evaluation_id=None,
        subnetwork_id=None,
        eps=1e-3,
        compress_outputs=None,
        indent=0,
        **kwargs  # Ignored
):
    assert isinstance(output_dir, str), 'output_dir must be given as a str'
    assert isinstance(grid_params, dict), 'grid_params must be given as a dict'
    assert isinstance(sample_id, str), 'sample_id must be given as a str'
    assert isinstance(alignment_id, str), 'alignment_id must be given as a str'
    assert isinstance(aggregation_id, str), 'aggregation_id must be given as a str'

    t0 = time.time()
    print('%sAggregating grid' % (' ' * (indent * 2)))
    indent += 1

    kwargs = dict(
        output_dir=output_dir,
        grid_params=grid_params,
        sample_id=sample_id,
        alignment_id=alignment_id,
        evaluation_id=evaluation_id,
        aggregation_id=aggregation_id,
        subnetwork_id=subnetwork_id,
        eps=eps,
        compress_outputs=compress_outputs
    )

    aggregation_dir = get_path(output_dir, 'subdir', 'aggregate', aggregation_id)
    if not os.path.exists(aggregation_dir):
        os.makedirs(aggregation_dir)
    kwargs_path = get_path(output_dir, 'kwargs', 'aggregate', aggregation_id)
    with open(kwargs_path, 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'aggregate', aggregation_id, compressed=compress_outputs)
    evaluation_path = get_path(output_dir, 'evaluation', 'aggregate', aggregation_id)

    grid_settings = process_grid_params(grid_params)
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
    results = pd.concat(results, axis=0)

    # Select configuration
    best_ix = np.argmax(scores)
    best_id = grid_ids[best_ix]
    best_score = scores[best_ix]

    results['selected'] = (results.grid_id == best_id) & (results.parcel_type != 'baseline')
    results.to_csv(evaluation_path, index=False)

    # Save configuration
    best_grid_dir = get_path(output_dir, 'subdir', 'grid', best_id)
    sample_kwargs_path = get_path(best_grid_dir, 'kwargs', 'sample', sample_id)
    sample_kwargs_path_exists = os.path.exists(sample_kwargs_path)
    align_kwargs_path = get_path(best_grid_dir, 'kwargs', 'align', alignment_id)
    align_kwargs_path_exists = os.path.exists(align_kwargs_path)
    if evaluation_id is not None:
        evaluate_kwargs_path = get_path(best_grid_dir, 'kwargs', 'evaluate', evaluation_id)
        evaluate_kwargs_path_exists = os.path.exists(evaluate_kwargs_path)
    else:
        evaluate_kwargs_path = None
        evaluate_kwargs_path_exists = None
    assert sample_kwargs_path_exists, '%s does not exist' % sample_kwargs_path
    assert align_kwargs_path_exists or evaluate_kwargs_path_exists, ('Either %s or %s must exist' %
        (align_kwargs_path, evaluate_kwargs_path))
    assert evaluation_id is None or evaluate_kwargs_path_exists, '%s does not exist' % evaluate_kwargs_path

    sample_kwargs = get_cfg(sample_kwargs_path)
    align_kwargs = get_cfg(align_kwargs_path)
    if evaluation_id:
        evaluate_kwargs = get_cfg(evaluate_kwargs_path)
    else:
        evaluate_kwargs = None

    parcellate_kwargs = dict(
        output_dir=output_dir,
        action_sequence=None,
        sample_kwargs=sample_kwargs,
        align_kwargs=align_kwargs,
        evaluate_kwargs=evaluate_kwargs,
        aggregate_kwargs=None,
        grid_params=None,
        eps=eps,
    )

    with open(output_path, 'w') as f:
        yaml.safe_dump(parcellate_kwargs, f, sort_keys=False)

    print('%sBest grid_id: %s | atlas score: %0.3f' % (' ' * (indent * 2), best_id, best_score))

    print('%sAggregation time: %ds' % (' ' * (indent * 2), time.time() - t0))

    return parcellate_kwargs


def parcellate(
        output_dir,
        action_sequence,
        sample_kwargs,
        align_kwargs,
        evaluate_kwargs=None,
        aggregate_kwargs=None,
        grid_params=None,
        eps=1e-3,
        compress_outputs=True,
        overwrite=False,
        indent=0
):
    assert isinstance(output_dir, str), 'output_dir is required, must be given as a str'
    assert isinstance(action_sequence, list), 'action_sequence is required, must be given as a list of dict'
    assert isinstance(sample_kwargs, dict), 'sample_kwargs is required, must be given as a dict.'
    assert isinstance(align_kwargs, dict), 'sample_kwargs is required, must be given as a dict.'
    assert isinstance(aggregate_kwargs, dict) == isinstance(grid_params, dict), ('Either both aggregation_kwargs '
        'and grid_params must be provided as dicts, or neither can be.')

    t0 = time.time()
    print('%sParcellating' % (' ' * (indent * 2)))
    indent += 1

    kwargs = dict(
        output_dir=output_dir,
        action_sequence=action_sequence,
        sample_kwargs=sample_kwargs,
        align_kwargs=align_kwargs,
        evaluate_kwargs=evaluate_kwargs,
        aggregate_kwargs=aggregate_kwargs,
        grid_params=grid_params,
        eps=eps,
        compress_outputs=compress_outputs,
    )

    sample_id = get_action('sample', action_sequence)['id']
    alignment_id = get_action('align', action_sequence)['id']
    evaluation_id = get_action('evaluate', action_sequence)['id']
    aggregation_id = get_action('aggregate', action_sequence)['id']
    parcellation_id = get_action('parcellate', action_sequence)['id']

    assert isinstance(sample_id, str), 'sample_id is required, must be given as a str.'
    assert isinstance(alignment_id, str), 'alignment_id is required, must be given as a str.'
    assert isinstance(parcellation_id, str), 'parcellation_id is required, must be given as a str.'

    parcellation_dir = get_path(output_dir, 'subdir', 'parcellate', parcellation_id)
    if not os.path.exists(parcellation_dir):
        os.makedirs(parcellation_dir)
    kwargs_path = get_path(output_dir, 'kwargs', 'parcellate', parcellation_id)
    with open(kwargs_path, 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'parcellate', parcellation_id, compressed=compress_outputs)

    for _kwargs in (sample_kwargs, align_kwargs, evaluate_kwargs):
        if _kwargs is not None:
            _kwargs['output_dir'] = output_dir
            _kwargs['compress_outputs'] = compress_outputs

    suffix = get_suffix(compress_outputs)

    # Grid search
    if grid_params:
        print('%sGrid searching' % (' ' * (indent * 2)))

        _action_sequence = [x for x in action_sequence if x['type'] != 'aggregate']
        indent += 1
        grid_settings = process_grid_params(grid_params)
        for grid_setting in grid_settings:
            grid_id = get_grid_id(grid_setting)

            print('%sGrid id: %s' % (' ' * (indent * 2), grid_id))

            _output_dir = get_path(output_dir, 'subdir', 'grid', grid_id)
            _sample_kwargs = copy.deepcopy(sample_kwargs)
            _sample_kwargs.update(grid_setting)
            _align_kwargs = copy.deepcopy(align_kwargs)
            _sample_kwargs.update(grid_setting)
            _evaluate_kwargs = copy.deepcopy(evaluate_kwargs)
            if _evaluate_kwargs is not None:
                _evaluate_kwargs.update(grid_setting)

            # Recursion bottoms out since grid_params is None
            parcellate(
                _output_dir,
                _action_sequence,
                _sample_kwargs,
                align_kwargs=_align_kwargs,
                evaluate_kwargs=_evaluate_kwargs,
                eps=eps,
                compress_outputs=compress_outputs,
                overwrite=overwrite,
                indent=indent + 1
            )

        indent -= 1

        mtime, exists = check_deps(
            output_dir,
            action_sequence[1:],  # aggregate is always the 2nd entry if used
            compressed=True
        )
        stale = mtime == 1
        if overwrite or stale or not exists:
            # Recursion bottoms out since grid_params is None
            _aggregate_kwargs = copy.deepcopy(aggregate_kwargs)
            _action_sequence.append(get_action('aggregate', action_sequence))
            _aggregate_kwargs.update(dict(
                output_dir=output_dir,
                action_sequence=_action_sequence,
                grid_params=grid_params,
                sample_id=sample_id,
                alignment_id=alignment_id,
                evaluation_id=evaluation_id,
                aggregation_id=aggregation_id,
                eps=eps,
                compress_outputs=compress_outputs,
                indent=indent,
            ))
            parcellate_kwargs = aggregate(**_aggregate_kwargs)
        else:
            print('%sAggregation exists. Skipping. To re-aggregate, run with overwrite=True.' %
                  (' ' * (indent * 2)))
            aggregation_output_path = get_path(output_dir, 'output', 'aggregate', aggregation_id)
            with open(aggregation_output_path, 'r') as f:
                parcellate_kwargs = yaml.safe_load(f)

        parcellate_kwargs.update(dict(
            action_sequence=_action_sequence,
            indent=indent
        ))
        parcellate(**parcellate_kwargs)
    else:
        # Sample
        _kwargs = copy.deepcopy(sample_kwargs)
        _kwargs.update(dict(
            output_dir=output_dir,
            sample_id=sample_id
        ))
        _action_sequence = action_sequence[-1:]  # sample is always last
        mtime, exists = check_deps(
            output_dir,
            _action_sequence,
            compressed=True
        )
        stale = mtime == 1
        if overwrite or stale or not exists:
            sample(**_kwargs, indent=indent)
        else:
            print('%sSample exists. Skipping. To resample, run with overwrite=True.' %
                  (' ' * (indent * 2)))

        # Align
        _kwargs = copy.deepcopy(align_kwargs)
        _kwargs.update(dict(
            output_dir=output_dir,
            alignment_id=alignment_id
        ))
        _action_sequence = action_sequence[-2:]  # align is always 2nd to last
        mtime, exists = check_deps(
            output_dir,
            _action_sequence,
            compressed=True
        )
        stale = mtime == 1
        if overwrite or stale or not exists:
            align(**_kwargs, indent=indent)
        else:
            print('%sAlignment exists. Skipping. To re-align, run with overwrite=True.' %
                  (' ' * (indent * 2)))

        # Evaluate (optional)
        if evaluate_kwargs is not None:
            _kwargs = copy.deepcopy(evaluate_kwargs)
            _kwargs.update(dict(
                output_dir=output_dir,
                evaluation_id=evaluation_id
            ))
            _action_sequence = action_sequence[-3:]  # evaluate (if used) is always 3rd to last
            mtime, exists = check_deps(
                output_dir,
                _action_sequence,
                compressed=True
            )
            stale = mtime == 1
            if overwrite or stale or not exists:
                evaluate(**_kwargs, indent=indent)
            else:
                print('%sEvaluation exists. Skipping. To re-evaluate, run with overwrite=True.' %
                      (' ' * (indent * 2)))

        # Copy final files to destination
        results_copied = False
        if evaluation_id is not None:
            evaluation_dir = get_path(output_dir, 'subdir', 'evaluate', evaluation_id)
            for filename in os.listdir(evaluation_dir):
                if filename.endswith(suffix) or filename == PATHS['evaluate']['output']:
                    shutil.copy2(join(evaluation_dir, filename), join(parcellation_dir, filename))
                    if filename == PATHS['evaluate']['output']:
                        results_copied = True
        alignment_dir = get_path(output_dir, 'subdir', 'align', alignment_id)
        for filename in os.listdir(alignment_dir):
            if filename.endswith(suffix) or (not results_copied and filename == PATHS['align']['evaluation']):
                shutil.copy2(join(alignment_dir, filename), join(parcellation_dir, filename))

        assert os.path.exists(output_path)

    output = get_nii(output_path)

    print('%sTotal time elapsed: %ds' % (' ' * (indent * 2), time.time() - t0))

    return output










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
    row = dict(atlas_score=r)

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
        for p in (None, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            if p is None:
                _atlas = atlas
                suffix = ''
            else:
                _atlas = atlas > p
                suffix = '_atpgt%s' % p
            contrast = (_atlas * evaluation_atlas).sum() / _atlas.sum()
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
