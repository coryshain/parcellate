import shutil
import time
import datetime
import inspect
import yaml
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, FastICA
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
        detrend=False,
        standardize=True,
        independent_runs=False,
        data_fraction=1,
        tr=2,
        low_pass=0.1,
        high_pass=0.01,
        n_samples=100,
        n_components_pca=None,
        n_components_ica=None,
        use_connectivity_profile=False,
        clustering_kwargs=None,
        compress_outputs=True,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(sample_id, str), 'sample_id must be given as a str'
    assert 0 <= data_fraction <= 1, 'data_fraction must be a proportion between 0 and 1'

    t0 = time.time()

    stderr('%sSampling (sample_id=%s)\n' % (' ' * (indent * 2), sample_id))
    indent += 1

    assert isinstance(output_dir, str), 'output_dir must be provided'

    sample_dir = get_path(output_dir, 'subdir', 'sample', sample_id)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if clustering_kwargs is None:
        clustering_kwargs = dict(
            n_init=N_INIT,
            init_size=INIT_SIZE
        )

    if dump_kwargs:
        kwargs = dict(
            output_dir=output_dir,
            functional_paths=functional_paths,
            n_networks=n_networks,
            fwhm=fwhm,
            sample_id=sample_id,
            mask_path=mask_path,
            detrend=detrend,
            standardize=standardize,
            independent_runs=independent_runs,
            data_fraction=data_fraction,
            tr=tr,
            low_pass=low_pass,
            high_pass=high_pass,
            n_samples=n_samples,
            n_components_pca=n_components_pca,
            n_components_ica=n_components_ica,
            clustering_kwargs=clustering_kwargs,
            compress_outputs=compress_outputs
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'sample', sample_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'sample', sample_id, compressed=compress_outputs)

    input_data = InputData(
        functional_paths=functional_paths,
        fwhm=fwhm,
        mask_path=mask_path,
        standardize=standardize,
        detrend=detrend,
        tr=tr,
        low_pass=low_pass,
        high_pass=high_pass
    )
    v = input_data.v
    df = pd.DataFrame([dict(n_trs=input_data.n_trs, n_runs=input_data.n_runs)])
    metadata_path = get_path(output_dir, 'metadata', 'sample', sample_id)
    df.to_csv(metadata_path, index=False)

    n_runs = input_data.n_runs
    samples_all = []
    scores_all = []
    if independent_runs:
        timecourses = input_data.functionals
    else:
        timecourses = [input_data.timecourses]
    for i, timecourse in enumerate(timecourses):
        # Sample parcellations by clustering the voxel timecourses
        if n_networks > 256:
            dtype=np.uint16
        else:
            dtype=np.uint8
        samples = np.zeros((v, n_samples), dtype=dtype)  # Shape: <n_voxels, n_samples>
        scores = np.zeros(n_samples)  # Shape: <n_samples>
        t = timecourse.shape[-1]
        if use_connectivity_profile:
            A = standardize_array(timecourse)
            if data_fraction < 1:
                v_ = round(v * data_fraction)
                step = v // v_
                roi_ix = np.arange(0, v, step)
                B = A[roi_ix]
            else:
                B = A
            X = np.dot(
                A,
                B.T
            ) / t
            X = (X > np.quantile(X, 0.9)).astype(int)
        else:
            X = timecourse
        if n_components_pca:
            n_components = min(n_components_pca, t)
            m = PCA(n_components=n_components)
            X = m.fit_transform(X)
        if n_components_ica:
            n_components = min(n_components_ica, t)
            m = FastICA(n_components=n_components, whiten='unit-variance')
            X = m.fit_transform(X)
        for j in range(n_samples):
            if len(timecourses) > 1:
                suffix = ' for run %d/%d' % (i + 1, n_runs)
            else:
                suffix = ''
            if n_samples > 1:
                stderr('\r%sSample %d/%d%s' % (' ' * (indent * 2), j + 1, n_samples, suffix))
            m = MiniBatchKMeans(n_clusters=n_networks, **clustering_kwargs)
            _sample = m.fit_predict(X)
            samples[:, j] = _sample
            scores[j] = m.inertia_

        stderr('\n')
        samples_all.append(samples)
        scores_all.append(scores)
    samples = np.concatenate(samples_all, axis=-1)
    samples = input_data.unflatten(samples)
    samples.to_filename(output_path)
    scores = pd.DataFrame({'sample_score': np.concatenate(scores_all, axis=0)})
    evaluation_path = get_path(output_dir, 'evaluation', 'sample', sample_id, compressed=compress_outputs)
    scores.to_csv(evaluation_path, index=False)

    stderr('%sSampling time: %ds\n' % (' ' * (indent * 2), time.time() - t0))


def align(
        output_dir,
        alignment_id=None,
        sample_id=None,
        n_alignments=None,
        scoring_method='corr',
        atlas_threshold=None,
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
    scoring_method = scoring_method.lower()

    assert isinstance(output_dir, str), 'output_dir must be provided'

    alignment_dir = get_path(output_dir, 'subdir', 'align', alignment_id)
    if not os.path.exists(alignment_dir):
        os.makedirs(alignment_dir)
    if dump_kwargs:
        kwargs = dict(
            alignment_id=alignment_id,
            sample_id=sample_id,
            n_alignments=n_alignments,
            scoring_method=scoring_method,
            atlas_threshold=atlas_threshold,
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
    sample_path = get_path(output_dir, 'output', 'sample', sample_id, compressed=compress_outputs)
    assert os.path.exists(sample_path), 'Sample file %s not found' % sample_path

    data = Data(
        nii_ref_path=sample_path
    )
    sample_nii = data.nii_ref

    samples = data.flatten(sample_nii)
    samples = samples.T  # Shape: <n_samples, v>, values are integer network indices

    # We do a sparse alignment with slow python loops to avoid OOM for large n_samples or n_networks

    # Align to reference
    sample_scores = pd.read_csv(get_path(output_dir, 'evaluation', 'sample', sample_id))['sample_score'].values
    s_ix = np.argsort(sample_scores)
    samples = samples[s_ix]
    parcellation = align_samples(
        samples,
        scoring_method=scoring_method,
        n_alignments=n_alignments,
        indent=indent + 1
    )

    parcellation = data.unflatten(parcellation.T)
    parcellation.to_filename(output_path)

    stderr('%sAlignment time: %ds\n' % (' ' * (indent * 2), time.time() - t0))


def label(
        output_dir,
        reference_atlases,
        labeling_id=None,
        alignment_id=None,
        sample_id=None,
        scoring_method='corr',
        atlas_threshold=None,
        max_subnetworks=None,
        minmax_normalize=True,
        use_poibin=True,
        eps=1e-3,
        compress_outputs=True,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(labeling_id, str), 'alignment_id must be given as a str'

    t0 = time.time()

    stderr('%sLabeling (labeling_id=%s)\n' % (' ' * (indent * 2), labeling_id))
    indent += 1
    scoring_method = scoring_method.lower()

    assert isinstance(output_dir, str), 'output_dir must be provided'

    labeling_dir = get_path(output_dir, 'subdir', 'label', labeling_id)
    if not os.path.exists(labeling_dir):
        os.makedirs(labeling_dir)
    if dump_kwargs:
        kwargs = dict(
            reference_atlases=reference_atlases,
            labeling_id=labeling_id,
            alignment_id=alignment_id,
            sample_id=sample_id,
            scoring_method=scoring_method,
            atlas_threshold=atlas_threshold,
            max_subnetworks=max_subnetworks,
            minmax_normalize=minmax_normalize,
            use_poibin=use_poibin,
            eps=eps,
            compress_outputs=compress_outputs,
            output_dir=output_dir,
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'label', labeling_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'label', labeling_id, compressed=compress_outputs)
    if alignment_id is not None:
        input_path = get_path(output_dir, 'output', 'align', alignment_id, compressed=compress_outputs)
        assert os.path.exists(input_path), 'Sample file %s not found' % input_path
        average_first = True
    elif sample_id is not None:
        input_path = get_path(output_dir, 'output', 'sample', sample_id, compressed=compress_outputs)
        assert os.path.exists(input_path), 'Sample file %s not found' % input_path
        average_first = False
    else:
        raise ValueError('Exactly one of sample_id or alignment_id must be provided')

    input_nii = image.smooth_img(input_path, None)

    reference_data = AtlasData(
        atlases=reference_atlases,
        resampling_target_nii=input_nii,
        compress_outputs=compress_outputs
    )
    reference_atlas_names = reference_data.atlas_names
    reference_atlases = reference_data.atlases
    v = reference_data.v
    reference_data.save_atlases(labeling_dir, prefix=REFERENCE_ATLAS_PREFIX)

    input_data = reference_data.flatten(input_nii)
    if average_first:
        n_networks = input_data.shape[-1]
        n_samples = None
    else:
        n_networks = int(input_data.max() + 1)
        n_samples = input_data.shape[-1]
    input_data = input_data.T  # Shape: <(n_networks | n_samples), v>, values are integer network indices
    if not max_subnetworks:
        max_subnetworks = n_networks

    # Find candidate network(s) for each reference
    n_reference_atlases = len(reference_atlas_names)
    reference_atlas_scores = np.full((n_reference_atlases,), -np.inf)
    candidates = {}
    results = []
    indent += 1
    for j, reference_atlas_name in enumerate(reference_atlas_names):
        stderr('%sAtlas: %s\n' % (' ' * (indent * 2), reference_atlas_name))
        reference_atlas = reference_atlases[reference_atlas_name]
        if atlas_threshold is not None:
            _reference_atlas = binarize_array(reference_atlas, threshold=atlas_threshold)
        else:
            _reference_atlas = reference_atlas
        if average_first:
            scores = np.zeros(n_networks)
            for ni in range(n_networks):
                if scoring_method == 'corr':
                    _score = np.corrcoef(input_data[ni], _reference_atlas)[0, 1]
                elif scoring_method == 'avg':
                    _score = np.dot(input_data[ni], _reference_atlas) / input_data[ni].sum()
                else:
                    raise ValueError('Unrecognized scoring method %s.' % scoring_method)
                scores[ni] = _score
            reference_ix = np.argsort(scores, axis=-1)[::-1]
        else:
            indent += 1
            stderr('%sDirectly aligning samples to reference\n' % (' ' * (indent * 2)))
            samples_relabeled = np.zeros_like(input_data)
            scores = np.zeros((n_samples, n_networks))
            if scoring_method == 'corr':
                _reference_atlas = standardize_array(_reference_atlas)
            indent += 1
            for si in range(n_samples):
                stderr('\r%sSample %d/%d' % (' ' * (indent * 2), si + 1, n_samples))
                networks = (input_data[si][None, ...] == np.arange(n_networks)[..., None]).astype(float)
                if scoring_method == 'corr':
                    _networks = standardize_array(networks)
                    _scores = np.dot(
                        _networks,
                        _reference_atlas.T
                    ) / v
                else:
                    num = np.dot(
                        networks,
                        _reference_atlas.T
                    )
                    denom = networks.sum(axis=-1)
                    denom[np.where(denom == 0)] = 1
                    _scores = num / denom
                sort_ix = np.argsort(_scores)[::-1]
                ranks = np.argsort(sort_ix)
                scores[si] = _scores[sort_ix]
                samples_relabeled[si] = ranks[input_data[si]]
            input_data = samples_relabeled
            reference_ix = np.arange(n_networks)
            stderr('\n')
            indent -= 2
        candidate = None
        r = -np.inf
        r_prev = -np.inf
        candidate_list = []
        candidate_scores = []
        indent += 1
        for ni in range(max_subnetworks):
            stderr('\r%sSubnetwork %d' % (' ' * (indent * 2), ni + 1))
            ix = reference_ix[ni]
            if average_first:
                _score = scores[ix]
            else:
                _score = np.tanh(np.arctanh(scores[:, ix] * (1 - 2 * eps) + eps).mean(axis=-1))
            candidate_scores.append(_score)
            _candidate = candidate
            if average_first:
                candidate = input_data[ix]
            else:
                candidate = input_data == ni
                weights = scores[:, ni:ni+1]
                weights = minmax_normalize_array(weights)
                candidate = (candidate * weights).sum(axis=0) / weights.sum()
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
            if scoring_method == 'corr':
                r = np.corrcoef(candidate, reference_atlas)[0, 1]
            elif scoring_method == 'avg':
                r = np.dot(candidate, reference_atlas) / candidate.sum()
            else:
                raise ValueError('Unrecognized scoring method %s.' % scoring_method)

            if r <= r_prev:
                candidate_list = candidate_list[:-1]
                candidate_scores = candidate_scores[:-1]
                candidate = _candidate
                r = r_prev
                break
            r_prev = r
        stderr('\n')
        indent -= 1

        reference_atlas_scores[j] = r
        candidate_list.insert(0, candidate)
        candidate_scores.insert(0, r)

        for s, candidate in enumerate(candidate_list):
            row = {
                'parcel': reference_atlas_name if s == 0 else '%s_sub%d' % (reference_atlas_name, s),
                '%sname' % REFERENCE_ATLAS_PREFIX: reference_atlas_name,
                '%sscore' % REFERENCE_ATLAS_PREFIX: candidate_scores[s]
            }
            if s == 0:
                row['parcel_type'] = 'network'
            else:
                row['parcel_type'] = 'subnetwork%d' % s
            if minmax_normalize:
                candidate = minmax_normalize_array(candidate)
                candidate_list[s] = candidate
            row['n_voxels'] = candidate.sum()
            results.append(row)
            candidate = reference_data.unflatten(candidate)
            if s == 0:
                suffix = ''
            else:
                suffix = '_sub%d' % s
            suffix += get_suffix(compress_outputs)
            candidate.to_filename(join(labeling_dir, '%s%s' % (reference_atlas_name, suffix)))

        candidates[reference_atlas_name] = candidate_list

    indent -= 1

    results = pd.DataFrame(results)
    results.to_csv(output_path, index=False)

    stderr('%sLabeling time: %ds\n' % (' ' * (indent * 2), time.time() - t0))


def evaluate(
        output_dir,
        evaluation_atlases=None,
        evaluation_map=None,
        evaluation_id=None,
        labeling_id=None,
        use_poibin=True,
        compress_outputs=True,
        dump_kwargs=True,
        indent=0
):
    assert isinstance(evaluation_id, str), 'evaluation_id must be given as a str'
    assert isinstance(labeling_id, str), 'alignment_id must be given as a str'

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
            evaluation_map=evaluation_map,
            evaluation_id=evaluation_id,
            labeling_id=labeling_id,
            use_poibin=use_poibin,
            compress_outputs=compress_outputs,
        )
        kwargs_path = get_path(output_dir, 'kwargs', 'evaluate', evaluation_id)
        with open(kwargs_path, 'w') as f:
            yaml.safe_dump(kwargs, f, sort_keys=False)
    output_path = get_path(output_dir, 'output', 'evaluate', evaluation_id, compressed=compress_outputs)

    suffix = get_suffix(compress_outputs)

    if evaluation_atlases is None:
        evaluation_atlases = {}

    # Collect references atlases and alignments
    labeling_dir = get_path(output_dir, 'subdir', 'label', labeling_id)
    label_kwargs = get_cfg(get_path(output_dir, 'kwargs', 'label', labeling_id))
    if evaluation_map is None:
        evaluation_map = {}
        for path in os.listdir(labeling_dir):
            if path.startswith(REFERENCE_ATLAS_PREFIX):
                reference_atlas_name = path[len(REFERENCE_ATLAS_PREFIX):-len(suffix)]
                evaluation_map[reference_atlas_name] = list(evaluation_atlases.keys())
    else:
        for path in os.listdir(labeling_dir):
            if path.startswith(REFERENCE_ATLAS_PREFIX):
                reference_atlas_name = path[len(REFERENCE_ATLAS_PREFIX):-len(suffix)]
                if reference_atlas_name not in evaluation_map:
                    evaluation_map[reference_atlas_name] = []

    reference_atlas_names = label_kwargs['reference_atlases']
    if isinstance(reference_atlas_names, str):
        if reference_atlas_names.lower() in ('all', 'all_reference'):
            reference_atlas_names = ALL_REFERENCE
        else:
            reference_atlas_names = [reference_atlas_names]
    reference_atlases = []
    candidates = {}
    resampling_target_nii = None
    for reference_atlas in reference_atlas_names:
        reference_atlas_path = join(labeling_dir, '%s%s%s' % (REFERENCE_ATLAS_PREFIX, reference_atlas, suffix))
        assert os.path.exists(reference_atlas_path), 'Reference atlas %s not found' % reference_atlas_path
        reference_atlases.append({reference_atlas: reference_atlas_path})

        for path in os.listdir(labeling_dir):
            if path.startswith(reference_atlas) and path.endswith(suffix):
                atlas_name = path[:-len(suffix)]
                atlas_name = re.sub('_sub\d+$', '', atlas_name)
                if atlas_name == reference_atlas:
                    trim = len(suffix)
                    name = path[:-trim]
                    path = join(labeling_dir, path)
                    if reference_atlas not in candidates:
                        candidates[reference_atlas] = {}
                    candidates[reference_atlas][name] = get_nii(path, add_to_cache=False)
                    if resampling_target_nii is None:
                        resampling_target_nii = candidates[reference_atlas][name]

    # Format data
    reference_data = AtlasData(
        atlases=reference_atlases,
        resampling_target_nii=resampling_target_nii,
        compress_outputs=compress_outputs
    )
    reference_atlases = reference_data.atlases
    evaluation_data = AtlasData(
        atlases=evaluation_atlases,
        resampling_target_nii=resampling_target_nii,
        compress_outputs=compress_outputs
    )
    evaluation_atlases = evaluation_data.atlases
    evaluation_data.save_atlases(evaluation_dir, prefix=EVALUATION_ATLAS_PREFIX)
    for x in candidates:
        for y in candidates[x]:
            candidates[x][y] = reference_data.flatten(candidates[x][y])

    stderr(' ' * (indent * 2) + 'Results:\n')
    results = []
    for reference_atlas_name in reference_atlas_names:
        # Score reference atlas as if it were a candidate parcellation (baseline)
        reference_atlas = reference_atlases[reference_atlas_name]
        _evaluation_atlases = {x: evaluation_atlases[x] for x in evaluation_atlases
                               if x in evaluation_map[reference_atlas_name]}
        atlas = reference_atlas
        atlas_name = '%s%s' % (REFERENCE_ATLAS_PREFIX, reference_atlas_name)
        row = _get_evaluation_row(
            atlas,
            atlas_name,
            reference_atlas_name=reference_atlas_name,
            reference_atlases=reference_atlases,
            evaluation_atlases=_evaluation_atlases
        )
        row['parcel_type'] = 'baseline'
        results.append(row)
        stderr(_pretty_print_evaluation_row(row, indent=indent + 1) + '\n')

        # Score evaluation atlases as if they were candidate parcellations (baseline)
        for evaluation_atlas_name in _evaluation_atlases:
            atlas = _evaluation_atlases[evaluation_atlas_name]
            atlas_name = '%s%s' % (EVALUATION_ATLAS_PREFIX, evaluation_atlas_name)
            row = _get_evaluation_row(
                atlas,
                atlas_name,
                reference_atlas_name=reference_atlas_name,
                reference_atlases=reference_atlases,
                evaluation_atlases=_evaluation_atlases
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
                reference_atlas_name=reference_atlas_name,
                reference_atlases=reference_atlases,
                evaluation_atlases=_evaluation_atlases
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
        labeling_id=None,
        subnetwork_id=1,
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

    _labeling_id = get_action_attr('label', action_sequence, 'id')
    if labeling_id is None:
        labeling_id = _labeling_id
    else:
        assert labeling_id == _labeling_id, ('Mismatch between provided labeling_id (%s) '
            'and the one contained in action_sequence (%s).' % labeling_id, _labeling_id)

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
            evaluation_id=evaluation_id,
            aggregation_id=aggregation_id,
            labeling_id=labeling_id,
            subnetwork_id=subnetwork_id,
            kernel_radius=kernel_radius,
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
        labeling_dir = get_path(_output_dir, 'subdir', 'label', labeling_id)
        if os.path.exists(labeling_dir):
            results_file_path = get_path(_output_dir, 'output', 'label', labeling_id)
            _results = pd.read_csv(results_file_path)
            _results['grid_id'] = grid_id
            results.append(_results)
            score = _get_atlas_score_from_df(_results, subnetwork_id=subnetwork_id, eps=eps)
        else:
            raise ValueError(('No available selection criteria for grid_id %s (no evaluation or labeling '
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
            if action_type in ('sample', 'label', 'aggregate'):
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
    labeling_id = get_action_attr('label', action_sequence, 'id')
    aggregation_id = get_action_attr('aggregate', action_sequence, 'id')
    parcellation_id = get_action_attr('parcellate', action_sequence, 'id')

    assert isinstance(sample_id, str), 'sample_id is required, must be given as a str.'
    assert isinstance(labeling_id, str), 'labeling_id is required, must be given as a str.'
    assert isinstance(parcellation_id, str), 'parcellation_id is required, must be given as a str.'

    overwrite = get_overwrite(overwrite)

    use_grid = aggregation_id is not None

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
                    if action['type'] == 'parcellate':
                        # Don't pass down the top-level parcellation kwargs
                        _kwargs = {x: _kwargs[x] for x in _kwargs if x in
                                   ('output_dir', 'compress_outputs', 'parcellation_id')}
                        action['kwargs'] = _kwargs
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
        _action_sequence = []
        for action in action_sequence:
            if action['type'] == 'evaluate':
                # Evaluation is not a dependency of aggregate
                continue
            _action_sequence.append(action)
            if action['type'] == 'aggregate':
                break
        assert action is not None and action['type'] == 'aggregate', ('action type "aggregate" not found in '
            'action_sequence')
        mtime, exists = check_deps(
            output_dir,
            _action_sequence,
            compressed=True
        )
        stale = mtime == 1
        if overwrite['aggregate'] or stale or not exists:
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
        action = None
        _action_sequence = []
        parcellate_override_kwargs = get_action('parcellate', action_sequence)
        if parcellate_override_kwargs is not None and 'kwargs' in parcellate_override_kwargs:
            parcellate_override_kwargs = parcellate_override_kwargs['kwargs']
        for action in action_sequence:
            if action['type'] == 'aggregate':
                continue
            action_ = get_action(action['type'], parcellate_kwargs['action_sequence'])
            kwargs_update = {x: action['kwargs'][x] for x in action['kwargs'] if x not in grid_params}
            # Change any kwargs that differ between the config and the saved kwargs file
            if action['type'] != 'parcellate':
                _kwarg_keys = set(inspect.signature(ACTIONS[action['type']]).parameters.keys())
                kwargs_update = {x: kwargs_update[x] for x in kwargs_update if x in _kwarg_keys}
                action_['kwargs'].update(kwargs_update)
            # Change any kwargs that are over-ridden by the settings in the parcellate action
            if action['type'] in parcellate_override_kwargs:
                action_['kwargs'].update(parcellate_override_kwargs[action['type']])
        assert action is not None and action['type'] == 'parcellate', ('Final action type in '
            'grid searched "action_sequence" must be "parcellate"')
        action_prefix = []
        for action in action_sequence[:-1]:  # Add dependencies to grid, ignoring last ('parcellate') action
            if action['type'] != 'evaluate':  # Changing the evaluation doesn't make the aggregation stale
                action_prefix.append(dict(
                    type=action['type'],
                    id=action['id'],
                    kwargs={}
                ))
        parcellate_kwargs['action_sequence'] = action_prefix + parcellate_kwargs['action_sequence']
        parcellate_kwargs['dump_kwargs'] = False  # Don't let recursive call overwrite top-level kwargs file
        parcellate_kwargs['overwrite'] = overwrite
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
        labeling_id = get_action_attr('label', action_sequence, 'id')
        evaluation_id = get_action_attr('evaluate', action_sequence, 'id')
        aggregation_id = get_action_attr('aggregate', action_sequence_full, 'id')

        parcellate_kwargs = get_action_attr('parcellate', action_sequence, 'kwargs')

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
            if overwrite[action_type] or stale or not exists:
                do_action = True
            else:
                do_action = False

            if action_type == 'sample':
                if do_action:
                    action_kwargs.update(dict(
                        output_dir=output_dir,
                        sample_id=sample_id
                    ))
                    if action_type in parcellate_kwargs:
                        for key in parcellate_kwargs[action_type]:
                            action_kwargs[key] = parcellate_kwargs[action_type][key]
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
                    if action_type in parcellate_kwargs:
                        for key in parcellate_kwargs[action_type]:
                            action_kwargs[key] = parcellate_kwargs[action_type][key]
                    align(**action_kwargs, indent=indent)
                else:
                    stderr('%sAlignment exists. Skipping. To re-align, run with overwrite=True.\n' %
                          (' ' * (indent * 2)))
            elif action_type == 'label':
                if do_action:
                    action_kwargs.update(dict(
                        output_dir=output_dir,
                        labeling_id=labeling_id,
                        alignment_id=alignment_id,
                        sample_id=sample_id
                    ))
                    if action_type in parcellate_kwargs:
                        for key in parcellate_kwargs[action_type]:
                            action_kwargs[key] = parcellate_kwargs[action_type][key]
                    label(**action_kwargs, indent=indent)
                else:
                    stderr('%sLabeling exists. Skipping. To re-label, run with overwrite=True.\n' %
                          (' ' * (indent * 2)))
            elif action_type == 'evaluate':
                if do_action:
                    action_kwargs.update(dict(
                        output_dir=output_dir,
                        evaluation_id=evaluation_id,
                        labeling_id=labeling_id
                    ))
                    if action_type in parcellate_kwargs:
                        for key in parcellate_kwargs[action_type]:
                            action_kwargs[key] = parcellate_kwargs[action_type][key]
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

                    if alignment_id is not None:
                        alignment_dir = get_path(output_dir, 'subdir', 'align', alignment_id)
                        for filename in os.listdir(alignment_dir):
                            if filename.endswith(suffix):
                                shutil.copy(join(alignment_dir, filename), join(parcellation_dir, filename))

                    labeling_dir = get_path(output_dir, 'subdir', 'label', labeling_id)
                    for filename in os.listdir(labeling_dir):
                        if filename.endswith(suffix) or (not results_copied and filename == PATHS['label']['evaluation']):
                            shutil.copy(join(labeling_dir, filename), join(parcellation_dir, filename))
                else:
                    stderr('%sParcellation exists. Skipping. To re-parcellate, run with overwrite=True.\n' %
                          (' ' * (indent * 2)))
            else:
                raise ValueError('Unrecognized action_type %s' % action_type)

        with open(output_path, 'w') as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        assert os.path.exists(output_path)

    stderr('%sTotal time elapsed: %ds\n' % (' ' * (indent * 2), time.time() - t0))










######################################
#
#  PRIVATE HELPER METHODS
#
######################################


def _get_n_voxels(
        atlas,
):
    m, M = atlas.min(), atlas.max()
    if m < 0 or M > 1:
        atlas = minmax_normalize_array(np.clip(atlas, 0, np.inf))
    n_voxels = atlas.sum()

    row = dict(
        n_voxels=n_voxels
    )

    return row


def _get_atlas_score(
        atlas,
        reference_atlas,
):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.corrcoef(reference_atlas, atlas)[0, 1]

    row = {
        '%sscore' % REFERENCE_ATLAS_PREFIX: r
    }

    return row


def _get_evaluation_spcorr(
        atlas,
        evaluation_atlases,
):
    row = {}
    evaluation_atlas_names = list(evaluation_atlases.keys())
    for evaluation_atlas_name in evaluation_atlas_names:
        evaluation_atlas = evaluation_atlases[evaluation_atlas_name]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.corrcoef(atlas, evaluation_atlas)[0, 1]
        row['%s_score' % evaluation_atlas_name] = r

    return row


def _get_evaluation_contrasts(
        atlas,
        evaluation_atlases
):
    m, M = atlas.min(), atlas.max()
    if m < 0 or M > 1:
        atlas = minmax_normalize_array(np.clip(atlas, 0, np.inf))
    row = {}
    evaluation_atlas_names = list(evaluation_atlases.keys())
    for evaluation_atlas_name in evaluation_atlas_names:
        evaluation_atlas = evaluation_atlases[evaluation_atlas_name]
        denom = atlas.sum()
        if denom:
            contrast = (atlas * evaluation_atlas).sum() / denom
        else:
            contrast = 0
        row['%s_contrast' % evaluation_atlas_name] = contrast

    return row


def _get_evaluation_row(
        atlas,
        atlas_name,
        reference_atlas_name,
        reference_atlases=None,
        evaluation_atlases=None
):
    row = {
        'parcel': atlas_name,
        '%sname' % REFERENCE_ATLAS_PREFIX: reference_atlas_name
    }
    row.update(_get_n_voxels(atlas))
    if reference_atlases is not None:
        row.update(_get_atlas_score(
            atlas,
            reference_atlases[reference_atlas_name]
        ))
        _reference_atlases = {'%s%s' % (REFERENCE_ATLAS_PREFIX, x): reference_atlases[x] for x in reference_atlases}
        row.update(_get_evaluation_spcorr(
            atlas,
            _reference_atlases
        ))
    if evaluation_atlases is not None:
        _evaluation_atlases = {'%s%s' % (EVALUATION_ATLAS_PREFIX, x): evaluation_atlases[x] for x in evaluation_atlases}
        row.update(_get_evaluation_spcorr(
            atlas,
            _evaluation_atlases
        ))
        row.update(_get_evaluation_contrasts(
            atlas,
            _evaluation_atlases
        ))

    return row


def _pretty_print_evaluation_row(
        row,
        max_evals=None,
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
        elif (col == ('%sscore' % REFERENCE_ATLAS_PREFIX) or
                    (col.startswith(EVALUATION_ATLAS_PREFIX) and col.endswith('_score'))):
            _col = '_'.join(col.split('_')[:-1])
            if max_evals is None or len(scores) < max_evals:
                if col != 'ref_score':
                    scores.add(col)
                to_print.append('%s score: %0.3f' % (_col, row[col]))
        elif col.endswith('contrast'):
            _col = '_'.join(col.split('_')[:-1])
            if max_evals is None or len(contrasts) < max_evals:
                contrasts.add(col)
                to_print.append('%s contrast: %0.3f' % (_col, row[col]))

    to_print = ('\n%s' % (' ' * ((indent + 1) * 2))).join(to_print)
    to_print = ' ' * (indent * 2) + to_print

    return to_print


def _get_atlas_score_from_df(df_scores, subnetwork_id=None, eps=1e-3):
    reference_atlas_names = df_scores['%sname' % REFERENCE_ATLAS_PREFIX].unique().tolist()
    parcel_names = df_scores.parcel
    scores = df_scores['%sscore' % REFERENCE_ATLAS_PREFIX]
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
    label=label,
    evaluate=evaluate,
    aggregate=aggregate,
    parcellate=parcellate,
)
