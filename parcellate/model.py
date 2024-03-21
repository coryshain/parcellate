import os
import sys
import shutil
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


def parcellate(
        output_dir,
        n_networks,
        functional_paths,
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
    t0 = time.time()

    print('%sParcellating' % (' ' * (indent * 2)))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    kwargs = dict(
        output_dir=output_dir,
        n_networks=n_networks,
        functional_paths=functional_paths,
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

    if clustering_kwargs is None:
        clustering_kwargs = dict(
            n_init=N_INIT,
            init_size=INIT_SIZE
        )
        kwargs['clustering_kwargs'] = clustering_kwargs

    suffix = get_suffix(compress_outputs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, PARCELLATE_CFG_FILENAME), 'w') as f:
        yaml.dump(kwargs, f)

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
    parcellations = np.zeros((v, n_samples), dtype=dtype)  # Shape: <n_parcellations, n_networks, n_voxels>
    for i in range(n_samples):
        sys.stdout.write('\r%sSampling parcellation %d/%d' % (' ' * (indent * 2), i + 1, n_samples))
        sys.stdout.flush()
        m = MiniBatchKMeans(n_clusters=n_networks, **clustering_kwargs)
        parcellation = m.fit_predict(timecourses)
        parcellations[:, i] = parcellation
    sys.stdout.write('\n')
    sys.stdout.flush()
    parcellations = input_data.unflatten(parcellations)
    parcellations.to_filename(join(output_dir, '%s%s' % (SAMPLE_FILENAME_BASE, suffix)))

    print('%sParcellation time: %ds' % (' ' * (indent * 2), time.time() - t0))

    return parcellations


def align(
        output_dir,
        reference_atlases,
        alignment_id='main',
        max_subnetworks=None,
        minmax_normalize=True,
        use_poibin=True,
        eps=1e-3,
        compress_outputs=True,
        indent=0,
        **kwargs  # Ignored
):
    t0 = time.time()

    print('%sAligning (alignment_id=%s)' % (' ' * (indent * 2), alignment_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    kwargs = dict(
        reference_atlases=reference_atlases,
        alignment_id=alignment_id,
        max_subnetworks=max_subnetworks,
        minmax_normalize=minmax_normalize,
        use_poibin=use_poibin,
        eps=eps,
        compress_outputs=compress_outputs,
        output_dir=output_dir,
    )
    parcellations_path = join(output_dir, '%s.nii' % SAMPLE_FILENAME_BASE)
    if not os.path.exists(parcellations_path):
        parcellations_path = join(output_dir, '%s.nii.gz' % SAMPLE_FILENAME_BASE)
    assert os.path.exists(parcellations_path), 'Parcellations file %s not found' % parcellations_path


    alignment_subdir = '%s_%s' % (ALIGNMENT_SUBDIR, alignment_id)
    if basename(output_dir) != alignment_subdir:
        output_dir = join(output_dir, alignment_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, ALIGN_CFG_FILENAME), 'w') as f:
        yaml.dump(kwargs, f)

    reference_data = ReferenceData(
        reference_atlases=reference_atlases,
        compress_outputs=compress_outputs
    )
    reference_atlas_names = reference_data.reference_atlas_names
    reference_atlases = reference_data.reference_atlases
    v = reference_data.v
    reference_data.save_atlases(output_dir)

    parcellations = reference_data.flatten(image.smooth_img(parcellations_path, None))
    n_networks = int(parcellations.max() + 1)
    if not max_subnetworks:
        max_subnetworks = n_networks
    n_samples = parcellations.shape[-1]
    parcellations = parcellations.T  # Shape: <n_samples, v>, values are integer network indices

    # We do a sparse alignment with slow python loops to avoid OOM for large n_samples or n_networks

    # Rank samples by average best alignment to reference atlas(es)
    sample_scores = np.zeros(n_samples)
    _reference_atlases = np.stack(
        [reference_atlases[x] for x in reference_atlases],
        axis=0
    )
    _reference_atlases_z = standardize_array(_reference_atlases)
    for si in range(n_samples):
        s = parcellations[si][None, ...] == np.arange(n_networks)[..., None]
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
    parcellation = align_samples(parcellations, ref_ix, w=sample_scores)

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
            candidate.to_filename(join(output_dir, '%s%s' % (reference_atlas_name, suffix)))

        candidates[reference_atlas_name] = candidate_list

    results = pd.DataFrame(results)
    results.to_csv(join(output_dir, ALIGNMENT_EVALUATION_FILENAME), index=False)

    suffix = get_suffix(compress_outputs)
    _parcellation = reference_data.unflatten(parcellation.T)
    _parcellation.to_filename(join(output_dir, '%s%s' % (ALIGNMENT_FILENAME_BASE, suffix)))

    print('%sAlignment time: %ds' % (' ' * (indent * 2), time.time() - t0))

    return candidates, parcellation


def evaluate(
        output_dir,
        evaluation_atlases,
        evaluation_id='main',
        alignment_id='main',
        average_first=False,
        use_poibin=True,
        compress_outputs=True,
        indent=0,
        **kwargs  # Ignored
):
    t0 = time.time()

    print('%sEvaluating (evaluation_id=%s)' % (' ' * (indent * 2), evaluation_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    kwargs = dict(
        output_dir=output_dir,
        evaluation_atlases=evaluation_atlases,
        evaluation_id=evaluation_id,
        average_first=average_first,
        use_poibin=use_poibin,
        compress_outputs=compress_outputs,
    )

    # Collect references atlases and alignments
    alignment_subdir = '%s_%s' % (ALIGNMENT_SUBDIR, alignment_id)
    alignment_dir = join(output_dir, alignment_subdir)
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
                if path.endswith('.nii.gz') or path.endswith('.nii'):
                    if path.endswith('.nii.gz'):
                        trim = 7
                    else:
                        trim = 4
                    name = path[:-trim]
                    path = join(alignment_dir, path)
                    if reference_atlas not in candidates:
                        candidates[reference_atlas] = {}
                    candidates[reference_atlas][name] = get_nii(path)

    evaluation_subdir = '%s_%s' % (EVALUATION_SUBDIR, evaluation_id)
    output_dir = join(output_dir, evaluation_subdir)
    if basename(output_dir) != evaluation_subdir:
        output_dir = join(output_dir, evaluation_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, EVALUATE_CFG_FILENAME), 'w') as f:
        yaml.dump(kwargs, f)

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
    evaluation_data.save_atlases(output_dir)
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
    results.to_csv(join(output_dir, EVALUATION_FILENAME), index=False)

    print('%sEvaluation time: %ds' % (' ' * (indent * 2), time.time() - t0))


def run(
        parcellate_kwargs=None,
        align_kwargs=None,
        evaluate_kwargs=None,
        parcellation_id='main',
        alignment_id='main',
        evaluation_id='main',
        output_dir=None,
        overwrite=False,
        indent=0
):
    t0 = time.time()
    print('%sProcessing' % (' ' * (indent * 2)))
    indent += 1

    if output_dir is None:
        output_dir = get_val_from_kwargs(
            parcellate_kwargs=parcellate_kwargs,
            align_kwargs=align_kwargs,
            evaluate_kwargs=evaluate_kwargs,
        )
    else:
        for kwargs in (parcellate_kwargs, align_kwargs, evaluate_kwargs):
            kwargs['output_dir'] = output_dir

    if output_dir is None:
        print('No valid actions to run. Terminating.')
        return

    parcellation_subdir = '%s_%s' % (PARCELLATION_SUBDIR, parcellation_id)
    if basename(output_dir) != parcellation_subdir:
        output_dir = join(output_dir, parcellation_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    compress_outputs = parcellate_kwargs.get('compress_outputs', True)

    # Parcellate
    if parcellate_kwargs:
        _kwargs = parcellate_kwargs
        _kwargs['parcellation_id'] = parcellation_id
        mtime, exists = check_parcellation(output_dir, compressed=compress_outputs)
        stale = mtime == 1
        if overwrite or stale or not exists:
            parcellate(**_kwargs, indent=indent)
        else:
            print('%sParcellation exists. Skipping. To resample parcellation, run with ``overwrite=True``.' %
                  (' ' * ((indent + 1) * 2)))

    # Align
    if align_kwargs:
        _kwargs = align_kwargs
        _kwargs['alignment_id'] = alignment_id
        mtime, exists = check_alignment(output_dir, alignment_id, compressed=compress_outputs)
        stale = mtime == 1
        if overwrite or stale or not exists:
            align(**_kwargs, indent=indent)
        else:
            print('%sAlignment exists. Skipping. To re-align, run with ``overwrite=True``.' %
                  (' ' * ((indent + 1) * 2)))

    # Evaluate
    if evaluate_kwargs is not None:
        _kwargs = evaluate_kwargs
        _kwargs['evaluation_id'] = evaluation_id
        mtime, exists = check_evaluation(output_dir, alignment_id, evaluation_id, compressed=compress_outputs)
        stale = mtime == 1
        if overwrite or stale or not exists:
            evaluate(**_kwargs, indent=indent)
        else:
            print('%sEvaluation exists. Skipping. To re-evaluate, run with ``overwrite=True``.' %
                  (' ' * ((indent + 1) * 2)))

    print('%sTotal time elapsed: %ds' % (' ' * (indent * 2), time.time() - t0))










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

    assert set(target_parcels) == set(parcel_names), f'Parcel names and reference atlas names mismatch. ' + \
                                                     f'Parcel names: {parcel_names}. Reference atlas ' + \
                                                     f'names: {reference_atlas_names}'

    scores = scores[sel]
    score = np.tanh(np.arctanh(scores * (1 - 2 * eps) + eps).mean())

    return score










######################################
#
#  ENSEMBLE METHODS
#
######################################


def aggregate_grid(
        output_dir,
        grid_params=None,
        alignment_id='main',
        evaluation_id='main',
        aggregation_id='main',
        subnetwork_id=None,
        eps=1e-3,
        indent=0,
        **kwargs  # Ignored
):
    t0 = time.time()
    print('%sAggregating grid' % (' ' * (indent * 2)))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    kwargs = dict(
        output_dir=output_dir,
        grid_params=grid_params,
        alignment_id=alignment_id,
        evaluation_id=evaluation_id,
        aggregation_id=aggregation_id,
        subnetwork_id=subnetwork_id,
        eps=eps
    )

    # Descend into the grid search subdirectory
    if not basename(output_dir) == GRID_SUBDIR:
        output_dir = join(output_dir, GRID_SUBDIR)
    alignment_subdir = '%s_%s' % (ALIGNMENT_SUBDIR, alignment_id)
    evaluation_subdir = '%s_%s' % (EVALUATION_SUBDIR, evaluation_id)

    grid_settings = process_grid_params(grid_params)
    results = []
    grid_ids = []
    scores = []
    for grid_setting in grid_settings:
        grid_id = get_grid_id(grid_setting)
        grid_ids.append(grid_id)
        evaluation_dir = join(output_dir, grid_id, evaluation_subdir)
        if os.path.exists(evaluation_dir):
            results_file_path = join(evaluation_dir, EVALUATION_FILENAME)
            _results = pd.read_csv(results_file_path)
            _results['grid_id'] = grid_id
            results.append(_results)
            score = _get_atlas_score_from_df(_results, subnetwork_id=subnetwork_id, eps=eps)
        else:
            alignment_dir = join(output_dir, grid_id, alignment_subdir)
            if os.path.exists(alignment_dir):
                results_file_path = join(alignment_dir, ALIGNMENT_EVALUATION_FILENAME)
                _results = pd.read_csv(results_file_path)
                _results['grid_id'] = grid_id
                results.append(_results)
                score = _get_atlas_score_from_df(_results, subnetwork_id=subnetwork_id, eps=eps)
            else:
                raise ValueError(('No available selection criteria for grid_id %s (no alignment or evaluation '
                                  'data found). Aggregation failed.' % grid_id))
        scores.append(score)
    results = pd.concat(results, axis=0)

    # Ascend into the top-level directory
    output_dir = dirname(output_dir)

    # Save results
    aggregation_path = get_aggregation_path(output_dir, aggregation_id)
    output_dir = dirname(aggregation_path)
    evaluation_path = join(output_dir, AGGREGATION_EVALUATION_FILENAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, AGGREGATE_CFG_FILENAME), 'w') as f:
        yaml.dump(kwargs, f)

    # Select configuration
    best_ix = np.argmax(scores)
    best_id = grid_ids[best_ix]
    best_score = scores[best_ix]

    results['selected'] = (results.grid_id == best_id) & (results.parcel_type != 'baseline')
    results.to_csv(evaluation_path, index=False)

    # Ascend into the top-level directory
    output_dir = dirname(output_dir)

    # Save configuration
    source_dir = join(output_dir, GRID_SUBDIR, best_id)
    parcellate_cfg_path = join(source_dir, PARCELLATE_CFG_FILENAME)
    parcellate_cfg_exists = os.path.exists(parcellate_cfg_path)
    align_cfg_path = join(dirname(get_alignment_path(source_dir, alignment_id)), ALIGN_CFG_FILENAME)
    align_cfg_exists = os.path.exists(align_cfg_path)
    evaluate_cfg_path = join(dirname(get_evaluation_path(source_dir, evaluation_id)), EVALUATE_CFG_FILENAME)
    evaluate_cfg_exists = os.path.exists(evaluate_cfg_path)
    assert parcellate_cfg_exists, '%s does not exist' % parcellate_cfg_path
    assert (align_cfg_exists or evaluate_cfg_exists), 'Either %s or %s must exist' % (align_cfg_path, evaluate_cfg_path)

    kwargs = dict(
        grid_id=best_id,
        parcellate=get_cfg(parcellate_cfg_path)
    )
    if align_cfg_exists:
        cfg = get_cfg(align_cfg_path)
        alignment_id = cfg.pop('alignment_id')
        kwargs['align'] = {alignment_id: cfg}
    if evaluate_cfg_exists:
        cfg = get_cfg(evaluate_cfg_path)
        evaluation_id = cfg.pop('evaluation_id')
        kwargs['evaluate'] = {evaluation_id: cfg}
    with open(join(dirname(aggregation_path), AGGREGATION_FILENAME), 'w') as f:
        yaml.dump(kwargs, f)

    print('%sBest grid_id: %s | atlas score: %0.3f' % (' ' * (indent * 2), best_id, best_score))

    print('%sAggregation time: %ds' % (' ' * (indent * 2), time.time() - t0))

    return kwargs


def run_grid(
        parcellate_kwargs,
        align_kwargs=None,
        evaluate_kwargs=None,
        aggregate_kwargs=None,
        refit_kwargs=None,
        grid_params=None,
        parcellation_id='main',
        alignment_id='main',
        evaluation_id='main',
        aggregation_id='main',
        refit=None,
        output_dir=None,
        compress_outputs=None,
        overwrite=False,
        indent=0
):
    t0 = time.time()
    print('%sGrid searching' % (' ' * (indent * 2)))
    indent += 1

    kwargs = dict(
        parcellate_kwargs=parcellate_kwargs,
        align_kwargs=align_kwargs,
        evaluate_kwargs=evaluate_kwargs,
        aggregate_kwargs=aggregate_kwargs,
        refit_kwargs=refit_kwargs,
        grid_params=grid_params,
        parcellation_id=parcellation_id,
        alignment_id=alignment_id,
        evaluation_id=evaluation_id,
        aggregation_id=aggregation_id,
        refit=refit,
        output_dir=output_dir,
    )

    if output_dir is None:
        output_dir = get_val_from_kwargs(
            'output_dir',
            parcellate_kwargs=parcellate_kwargs,
            align_kwargs=align_kwargs,
            evaluate_kwargs=evaluate_kwargs,
            aggregate_kwargs=aggregate_kwargs
        )
    else:
        for _kwargs in (parcellate_kwargs, align_kwargs, evaluate_kwargs):
            if _kwargs is not None:
                _kwargs['output_dir'] = output_dir
    if output_dir is None:
        print('No valid actions to run. Terminating.')
        return

    if compress_outputs is None:
        compress_outputs = get_val_from_kwargs(
            'compress_outputs',
            parcellate_kwargs=parcellate_kwargs,
            align_kwargs=align_kwargs,
            evaluate_kwargs=evaluate_kwargs,
            aggregate_kwargs=aggregate_kwargs
        )

    # Descend into the grid search subdirectory
    if not basename(output_dir) == GRID_SUBDIR:
        output_dir = join(output_dir, GRID_SUBDIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(output_dir, GRID_CFG_FILENAME), 'w') as f:
        yaml.dump(kwargs, f)

    if (
            parcellate_kwargs is not None or
            align_kwargs is not None or
            evaluate_kwargs is not None
    ):
        grid_settings = process_grid_params(grid_params)
        for grid_setting in grid_settings:
            grid_id = get_grid_id(grid_setting)
            _output_dir = output_dir
            _output_dir = join(_output_dir, grid_id)

            print('%sGrid id: %s' % (' ' * (indent * 2), grid_id))

            # Parcellate
            if parcellate_kwargs is not None:
                mtime, exists = check_parcellation(_output_dir, compressed=compress_outputs)
                stale = mtime == 1
                if overwrite or stale or not exists:
                    _kwargs = parcellate_kwargs.copy()
                    _kwargs.update(grid_setting)
                    _kwargs.update(dict(
                        output_dir=_output_dir
                    ))
                    parcellate(**_kwargs, indent=indent + 1)
                else:
                    print('%sParcellation exists. Skipping. To resample parcellation, run with ``overwrite=True``.' %
                          (' ' * ((indent + 1) * 2)))

            # Align
            if align_kwargs is not None:
                mtime, exists = check_alignment(_output_dir, alignment_id, compressed=compress_outputs)
                stale = mtime == 1
                if overwrite or stale or not exists:
                    _kwargs = align_kwargs.copy()
                    _kwargs.update(grid_setting)
                    _kwargs.update(dict(
                        alignment_id=alignment_id,
                        output_dir=_output_dir
                    ))
                    align(**_kwargs, indent=indent + 1)
                else:
                    print('%sAlignment exists. Skipping. To re-align, run with ``overwrite=True``.' %
                          (' ' * ((indent + 1) * 2)))

            # Evaluate
            if evaluate_kwargs is not None:
                mtime, exists = check_evaluation(_output_dir, alignment_id, evaluation_id, compressed=compress_outputs)
                stale = mtime == 1
                if overwrite or stale or not exists:
                    _kwargs = evaluate_kwargs.copy()
                    _kwargs.update(grid_setting)
                    _kwargs.update(dict(
                        evaluation_id=evaluation_id,
                        output_dir=_output_dir
                    ))
                    evaluate(**_kwargs, indent=indent + 1)
                else:
                    print('%sEvaluation exists. Skipping. To re-evaluate, run with ``overwrite=True``.' %
                          (' ' * ((indent + 1) * 2)))

    # Ascend into the top-level directory
    output_dir = dirname(output_dir)

    # Aggregate results and find winning parcellation
    aggregation_ran = False
    if aggregate_kwargs is not None:

        mtime, exists = check_aggregation(
            output_dir,
            alignment_id,
            evaluation_id,
            aggregation_id,
            grid_params,
            compressed=compress_outputs
        )
        stale = mtime == 1
        if overwrite or stale or not exists:
            _kwargs = aggregate_kwargs.copy()
            _kwargs.update(dict(
                output_dir=output_dir,
                grid_params=grid_params,
                alignment_id=alignment_id,
                evaluation_id=evaluation_id,
                aggregation_id=aggregation_id,
                indent=indent,
            ))
            aggregate_grid(**_kwargs)
            aggregation_ran = True
        else:
            print('%sAggregation exists. Skipping. To re-aggregate, run with ``overwrite=True``.' %
                  (' ' * ((indent + 1) * 2)))

    # Copy or refit winning parcellation
    if refit_kwargs is not None:
        aggregation_path = get_aggregation_path(output_dir, aggregation_id)
        cfg = get_cfg(aggregation_path)

        parcellation_subdir = '%s_%s' % (PARCELLATION_SUBDIR, parcellation_id)
        if basename(output_dir) != parcellation_subdir:
            output_dir = join(output_dir, parcellation_subdir)

        do_refit = aggregation_ran  # Aggregation just ran, refit
        if not do_refit:
            mtime = get_parcellation_mtime(output_dir, compressed=compress_outputs)
            if mtime is None:  # Target parcellation doesn't exist, refit
                do_refit = True
            else:
                aggregation_mtime, _ = check_aggregation(
                    dirname(output_dir),
                    alignment_id,
                    evaluation_id,
                    aggregation_id,
                    grid_params,
                    compressed=compress_outputs
                )
                if aggregation_mtime and mtime < aggregation_mtime:  # Target parcellation is stale, refit
                    do_refit = True

        if do_refit:
            print('%sRefitting' % (' ' * (indent * 2)))
            _parcellate_kwargs = get_parcellate_kwargs(cfg)
            _parcellate_kwargs.update(refit_kwargs)
            _align_kwargs = get_align_kwargs(cfg, alignment_id)
            _evaluate_kwargs = get_evaluate_kwargs(cfg, evaluation_id)

            run(
                parcellate_kwargs=_parcellate_kwargs,
                align_kwargs=_align_kwargs,
                evaluate_kwargs=_evaluate_kwargs,
                parcellation_id=parcellation_id,
                alignment_id=alignment_id,
                evaluation_id=evaluation_id,
                output_dir=output_dir,
                overwrite=True,
                indent=indent + 1
            )

    elif aggregate_kwargs is not None:
        if aggregation_ran:  # Aggregation was performed and no refitting is requested, so copy the winner
            aggregation_path = get_aggregation_path(output_dir, aggregation_id)
            aggregation_cfg = get_cfg(aggregation_path)
            grid_id = aggregation_cfg['grid_id']
            winner_path = os.path.join(output_dir, GRID_SUBDIR, grid_id)
            out_path = '%s_%s' % (PARCELLATION_SUBDIR, parcellation_id)
            if basename(output_dir) != out_path:
                out_path = join(output_dir, out_path)
            shutil.copytree(winner_path, out_path, dirs_exist_ok=True)

    print('%sGrid search total time elapsed: %ds' % (' ' * (indent * 2), time.time() - t0))
