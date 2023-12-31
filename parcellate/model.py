import sys
import os
import shutil
import time
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn()
# except ModuleNotFoundError:
#     pass
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.cluster import MiniBatchKMeans
from nilearn import image

from parcellate.data import ParcellateData


N_INIT = 1
INIT_SIZE = None
SEARCH_RESULTS_SUBDIR = 'search_results'
FINAL_PARCELLATION_SUBDIR = 'parcellation'
CONFIG_FILENAME = 'config.yml'
BASELINE_SCORES_FILENAME = 'baseline_scores.csv'
K_STR = 'k%03d'
K_SCORES_FILENAME = 'scores_by_k.csv'
ENSEMBLE_STR = 'e%03d'
ENSEMBLE_SCORES_FILENAME = 'scores_by_ensemble_id.csv'


def parcellate(
        functionals,
        mask=None,
        standardize=True,
        normalize=False,
        detrend = False,
        tr=2,
        low_pass=0.1,
        high_pass=0.01,
        reference_atlases=None,
        evaluation_atlases=None,
        atlas_lower_cutoff=None,
        atlas_upper_cutoff=None,
        n_networks=50,
        max_networks=None,
        n_samples=100,
        n_ensemble=1,
        align_to_reference=False,
        clustering_kwargs=None,
        eps=1e-3,
        minmax=True,
        output_dir='parcellation_output',
        dump_config_to_output_dir=True,
        overwrite=False
):
    kwargs = dict(
        functionals=functionals,
        mask=mask,
        standardize=standardize,
        normalize=normalize,
        detrend=detrend,
        tr=tr,
        low_pass=low_pass,
        high_pass=high_pass,
        reference_atlases=reference_atlases,
        evaluation_atlases=evaluation_atlases,
        atlas_lower_cutoff=atlas_lower_cutoff,
        atlas_upper_cutoff=atlas_upper_cutoff,
        n_networks=n_networks,
        max_networks=max_networks,
        n_samples=n_samples,
        n_ensemble=n_ensemble,
        align_to_reference=align_to_reference,
        clustering_kwargs=clustering_kwargs,
        eps=eps,
        minmax=minmax,
        output_dir=output_dir,
        dump_config_to_output_dir=dump_config_to_output_dir
    )

    T0 = time.time()

    if max_networks is None:
        max_networks = n_networks
    assert n_networks <= max_networks, 'n_networks (%d) must be <= max_networks (%d)' % (n_networks, max_networks)

    if clustering_kwargs is None:
        clustering_kwargs = dict(
            n_init=N_INIT,
            init_size=INIT_SIZE
        )

    if n_ensemble is None:
        n_ensemble = 1
    try:
        n_ensemble = int(n_ensemble)
    except ValueError:
        raise ValueError('n_ensemble must be type ``int`` or ``None``. Got %s.' % type(n_ensemble))

    print('  Loading data')
    t0 = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if dump_config_to_output_dir:
        with open(os.path.join(output_dir, CONFIG_FILENAME), 'w') as f:
            yaml.dump(kwargs, f)

    data = ParcellateData(
        functionals=functionals,
        mask=mask,
        standardize=standardize,
        normalize=normalize,
        detrend=detrend,
        tr=tr,
        low_pass=low_pass,
        high_pass=high_pass,
        reference_atlases=reference_atlases,
        evaluation_atlases=evaluation_atlases,
        atlas_lower_cutoff=atlas_lower_cutoff,
        atlas_upper_cutoff=atlas_upper_cutoff
    )

    data.save_atlases(output_dir)

    v = data.v
    reference_atlases = data.reference_atlases
    evaluation_atlases = data.evaluation_atlases

    print('    Time: %ds' % (time.time() - t0))
    print('  N voxels: %d' % v)

    baseline_scores = {}
    for reference_atlas_name in reference_atlases:
        reference_atlas = reference_atlases[reference_atlas_name]
        if reference_atlas_name in evaluation_atlases:
            print('  Baseline evaluations for reference atlas %s:' % reference_atlas_name)
            _evaluation_atlases = evaluation_atlases[reference_atlas_name]
            evaluation_atlas_names = list(_evaluation_atlases.keys())
            for evaluation_atlas_name in evaluation_atlas_names:
                evaluation_atlas = _evaluation_atlases[evaluation_atlas_name]
                r = np.corrcoef(reference_atlas, evaluation_atlas)[0, 1]
                baseline_scores['%s_v_%s' % (reference_atlas_name, evaluation_atlas_name)] = r
                print('    %s atlas vs. %s score: %.3f' % (reference_atlas_name, evaluation_atlas_name, r))
            for i in range(len(evaluation_atlas_names)):
                for j in range(i + 1, len(evaluation_atlas_names)):
                    name1 = evaluation_atlas_names[i]
                    atlas1 = _evaluation_atlases[name1]
                    name2 = evaluation_atlas_names[j]
                    atlas2 = _evaluation_atlases[name2]
                    r = np.corrcoef(atlas1, atlas2)[0, 1]
                    baseline_scores['%s_v_%s' % (name1, name2)] = r
                    print('    %s vs. %s score: %.3f' % (name1, name2, r))
    baseline_scores = pd.DataFrame([baseline_scores])
    baseline_scores.to_csv(os.path.join(output_dir, BASELINE_SCORES_FILENAME), index=False)

    k_scores_path = os.path.join(output_dir, K_SCORES_FILENAME)
    if os.path.exists(k_scores_path):
        k_scores = pd.read_csv(k_scores_path, index_col='index')
    else:
        k_scores = None
    if len(reference_atlases):
        k_score_col = 'reference_atlas_score'
    else:
        k_score_col = 'inter_network_spcorr'
    for k in range(n_networks, max_networks + 1):
        if k_scores is None:
            is_done = False
        else:
            try:
                k_scores.loc[k]
                is_done = True
            except KeyError:
                is_done = False

        if not is_done or overwrite:
            t0 = time.time()

            data_row = parcellate_k(
                k,
                data,
                n_samples=n_samples,
                align_to_reference=align_to_reference,
                clustering_kwargs=clustering_kwargs,
                eps=eps,
                minmax=minmax,
                ensemble_id=0,
                output_dir=os.path.join(output_dir, SEARCH_RESULTS_SUBDIR)
            )

            if k_scores is None:
                k_scores = pd.DataFrame(columns=list(data_row.keys()))
            k_scores.loc[k] = data_row
            k_scores.reset_index().to_csv(k_scores_path, index=False)

            print('    Time: %ds' % (time.time() - t0))

    best_k_ix = np.argmax(k_scores[k_score_col].values)
    best_k_row = k_scores.iloc[best_k_ix]
    best_k, best_k_score = best_k_row['k'], best_k_row[k_score_col]

    print('Best k: %d, score: %0.3f' % (best_k, best_k_score))

    k = best_k

    if n_ensemble and n_ensemble > 1:
        ensemble_scores_path = os.path.join(output_dir, ENSEMBLE_SCORES_FILENAME)
        if os.path.exists(ensemble_scores_path):
            ensemble_scores = pd.read_csv(ensemble_scores_path, index_col='index')
            zeroth_row = k_scores[k_scores['k'] == k].iloc[0]
            zeroth_row['ensemble_id'] = 0
            ensemble_scores.loc[0] = zeroth_row
        else:
            ensemble_scores = pd.DataFrame(k_scores[k_scores['k'] == k])
            ensemble_scores['ensemble_id'] = [0]
            ensemble_scores = ensemble_scores.reset_index(drop=True)
        for ensemble_id in range(1, n_ensemble):
            try:
                ensemble_scores.loc[ensemble_id]
                is_done = True
            except KeyError:
                is_done = False

            if not is_done or overwrite:
                print('  Sampling ensemble component %d' % ensemble_id)
                t0 = time.time()

                data_row = parcellate_k(
                    k,
                    data,
                    n_samples=n_samples,
                    align_to_reference=align_to_reference,
                    clustering_kwargs=clustering_kwargs,
                    eps=eps,
                    minmax=minmax,
                    ensemble_id=ensemble_id,
                    output_dir=os.path.join(output_dir, SEARCH_RESULTS_SUBDIR)
                )
                data_row['ensemble_id'] = ensemble_id
                ensemble_scores.loc[ensemble_id] = data_row
                ensemble_scores.reset_index().to_csv(ensemble_scores_path, index=False)

                print('    Time: %ds' % (time.time() - t0))

        best_ensemble_ix = np.argmax(ensemble_scores[k_score_col].values)
        best_ensemble_row = ensemble_scores.iloc[best_ensemble_ix]
        best_ensemble, best_ensemble_score = best_ensemble_row['ensemble_id'], best_ensemble_row[k_score_col]

        print('Best ensemble component: %d, score: %0.3f' % (best_ensemble, best_ensemble_score))
        selected_subdir = (K_STR % k) + '_' + (ENSEMBLE_STR % best_ensemble)
    else:
        selected_subdir = (K_STR % k) + '_' + (ENSEMBLE_STR % 0)

    final_dir_path = os.path.join(output_dir, FINAL_PARCELLATION_SUBDIR)
    if os.path.exists(final_dir_path):
        shutil.rmtree(final_dir_path)

    shutil.copytree(
        os.path.join(output_dir, SEARCH_RESULTS_SUBDIR, selected_subdir),
        final_dir_path
    )

    print('Total time elapsed: %ds' % (time.time() - T0))
    print('Done')


def parcellate_k(
        k,
        data,
        n_samples=100,
        align_to_reference=False,
        clustering_kwargs=None,
        eps=1e-3,
        minmax=True,
        ensemble_id=None,
        output_dir='parcellation_output',
        suffix=''
):
    if ensemble_id is None:
        ensemble_id = 0
    print('  Parcellating (k = %d)' % k)
    k_str = (K_STR % k) + '_' + (ENSEMBLE_STR % ensemble_id)
    results_dir = os.path.join(output_dir, k_str)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    v = data.v
    timecourses = data.timecourses
    reference_atlases = data.reference_atlases
    reference_atlas_names = data.reference_atlas_names
    evaluation_atlases = data.evaluation_atlases

    # Sample parcellations by clustering the voxel timecourses
    parcellations = np.zeros((n_samples, k, v))  # Shape: <n_parcellations, n_networks, n_voxels>
    parcellation_inertias = np.full((n_samples,), np.inf)
    for i in range(n_samples):
        sys.stdout.write('\r    Sampling parcellation %d/%d' % (i + 1, n_samples))
        sys.stdout.flush()
        m = MiniBatchKMeans(n_clusters=k, **clustering_kwargs)
        parcellation = m.fit_predict(timecourses)
        parcellation = label_binarize(parcellation, classes=np.arange(k)).astype('float32').T
        if k == 2:  # label_binarize collapses 2-class labels, so expand them
            parcellation = np.concatenate([parcellation, 1 - parcellation], axis=0)
        parcellation_inertia = m.inertia_

        parcellations[i] = parcellation
        parcellation_inertias[i] = parcellation_inertia
    print()

    # Normalize to facilitate spcorr computation
    parcellations_z = (parcellations - parcellations.mean(axis=-1, keepdims=True)) / \
                      parcellations.std(axis=-1, keepdims=True)

    # Networks in each parcellation are not yet aligned.
    # Now we align them, first with respect to any reference networks, and then with respect to the parcellation
    # with the lowest inertia.

    n_reference_atlases = len(reference_atlas_names)
    if align_to_reference:
        # Extract any target networks by comparison to the corresponding reference atlas
        reference_parcellations = np.zeros((n_samples, n_reference_atlases, v))
        for j, reference_atlas in enumerate(reference_atlas_names):
            # Find and extract the network with the highest correlation to the reference in each parcellation
            n_networks = parcellations.shape[1]
            reference_atlas = reference_atlases[reference_atlas]
            reference_scores = np.clip(np.dot(parcellations_z, reference_atlas) / v, -1, 1)  # Shape:
            # <n_parcellations,
            # n_networks>
            reference_alignments = np.argmax(reference_scores, axis=-1)
            parcellation = parcellations[np.arange(n_samples), reference_alignments]
            reference_parcellations[:, j] = parcellation

            # Remove the extracted network from each parcellation
            ix = reference_alignments[..., None] != np.arange(n_networks)[None, ...]
            parcellations = parcellations[ix].reshape(n_samples, n_networks - 1, v)
            parcellations_z = parcellations_z[ix].reshape(n_samples, n_networks - 1, v)
    else:
        reference_parcellations = None

    # Align all remaining networks of each parcellation to the best parcellation by maximizing the linear sum
    # assignment (correlation score)

    # Z-score the parcellations
    n_other_networks = parcellations.shape[1]
    # Find the reference parcellation (best clustering score)
    ref_ix = np.argmin(parcellation_inertias)
    # Compute correlations between each network of the reference parcellation and each network of every other
    # parcellation
    parcellations_spcorr = np.matmul(
        parcellations_z[ref_ix:ref_ix + 1, ...],
        np.transpose(parcellations_z, axes=(0, 2, 1))
    ) / v  # Shape: <n_parcellations - n_reference_atlases - 1, n_networks, n_networks>
    parcellations_spcorr = np.clip(parcellations_spcorr, -1, 1)
    # Align parcellations to the reference by finding the alignment that maximizes spcorr
    other_parcellations = np.zeros((n_samples, n_other_networks, v))
    for i in range(n_samples):
        if i == ref_ix:
            # Reference parcellation is already aligned with itself, just return it
            parcellation = parcellations[ref_ix]
        else:
            # Get the correlation matrix between reference and current parcellation (n networks x n networks)
            scores = parcellations_spcorr[i]
            # Find optimal alignment
            ix_l, ix_r = optimize.linear_sum_assignment(scores, maximize=True)
            # Get alignment scores for each network pair
            # Ensure sorting to match current network permutation of reference parcellation
            sort_ix = np.argsort(ix_l)
            ix_l, ix_r = ix_l[sort_ix], ix_r[sort_ix]
            # Align current parcelation to reference
            parcellation = parcellations[i, ix_r]
        other_parcellations[i] = parcellation
    # Combine reference and other parcellations
    if align_to_reference:
        parcellations = np.concatenate([
            reference_parcellations,
            other_parcellations
        ], axis=1)
    else:
        parcellations = other_parcellations
    parcellations_z = (parcellations - parcellations.mean(axis=-1, keepdims=True)) / \
                      parcellations.std(axis=-1, keepdims=True)
    # Now parcellations are aligned

    data_row = {'k': k}

    reference_atlas_scores = np.full((n_reference_atlases,), -np.inf)
    for j, reference_atlas_name in enumerate(reference_atlas_names):
        reference_atlas = reference_atlases[reference_atlas_name]
        if align_to_reference:
            atlases = reference_parcellations[:, j]
        else:
            scores = np.clip(np.dot(parcellations_z, reference_atlas) / v, -1, 1)
            reference_ix = np.argmax(scores, axis=-1)
            atlases = parcellations[np.arange(n_samples), reference_ix]

        r_all = np.tril(np.corrcoef(atlases), -1)
        r_mean = np.tanh(np.arctanh(r_all * (1 - 2 * eps) + eps).mean())
        data_row['%s_consistency_score' % reference_atlas_name] = r_mean

        atlas = atlases.mean(axis=0)
        if minmax:
            atlas = data.minmax_normalize(atlas)
        r = np.corrcoef(atlas, reference_atlas)[0, 1]
        data_row['%s_atlas_score' % reference_atlas_name] = r
        reference_atlas_scores[j] = r

        to_print = '    %s network | Consistency score: %.3f | Atlas score: %.3f' % (reference_atlas_name,
                                                                                     r_mean, r)

        if reference_atlas_name in evaluation_atlases:
            _evaluation_atlases = evaluation_atlases[reference_atlas_name]
            for evaluation_name in _evaluation_atlases:
                evaluation_atlas = _evaluation_atlases[evaluation_name]
                r = np.corrcoef(atlas, evaluation_atlas)[0, 1]
                data_row['%s_%s_score' % (reference_atlas_name, evaluation_name)] = r
                to_print += ' | %s score: %.3f' % (evaluation_name, r)

        network = data.unflatten(atlas)
        network.to_filename(os.path.join(results_dir, '%s_network%s.nii' % (reference_atlas_name, suffix)))

        print(to_print)

    for j in range(n_other_networks):
        atlases = other_parcellations[:, j]

        r_all = np.tril(np.corrcoef(atlases), -1)
        r_mean = np.tanh(np.arctanh(r_all * (1 - 2 * eps) + eps).mean())
        data_row['%03d_network_consistency_score' % (j + 1)] = r_mean
        atlas = atlases.mean(axis=0)
        if minmax:
            atlas = data.minmax_normalize(atlas)

        network = data.unflatten(atlas)
        network.to_filename(os.path.join(results_dir, '%03d_network%s.nii' % (j + 1, suffix)))

    _parcellations = parcellations.mean(axis=0)
    spcorr_networks = np.tril(np.corrcoef(_parcellations), -1)
    # Average with Fisher's method
    spcorr_networks = np.tanh(np.nanmean(np.arctanh(spcorr_networks * (1 - 2 * eps) + eps)))
    print('    Mean spcorr between networks: %.3f' % spcorr_networks)

    spcorr_parcellations = np.clip(
        np.matmul(parcellations_z, np.transpose(parcellations_z, axes=(0, 2, 1))) / v,
        -1,
        1
    )
    # Average with Fisher's method
    spcorr_parcellations = np.tanh(np.nanmean(np.arctanh(spcorr_parcellations * (1 - 2 * eps) + eps)))
    print('    Mean spcorr between parcellations: %.3f' % spcorr_parcellations)

    r = reference_atlas_scores
    reference_atlas_score = np.tanh(np.arctanh(r * (1 - 2 * eps) + eps).mean())

    data_row['inter_network_spcorr'] = spcorr_networks
    data_row['inter_parcellation_spcorr'] = spcorr_parcellations
    data_row['reference_atlas_score'] = reference_atlas_score
    data_row['timestamp'] = datetime.now()

    return data_row
