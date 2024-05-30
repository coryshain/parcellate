import sys
import os
import textwrap

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
import resources
import numpy as np
from scipy import stats, signal, optimize
from sklearn.preprocessing import normalize as sk_normalize
from nilearn import image, masking, plotting

from parcellate.util import REFERENCE_ATLAS_PREFIX, EVALUATION_ATLAS_PREFIX, join, get_suffix, stderr

NII_CACHE = {}  # Map from paths to NII objects


def standardize_array(arr, axis=-1):
    out = (arr - arr.mean(axis=axis, keepdims=True)) / arr.std(axis=axis, keepdims=True)
    out = np.where(np.isfinite(out), out, np.zeros_like(out))

    return out


def binarize_array(arr, threshold=0.):
    out = (arr > threshold).astype(arr.dtype)

    return out


def detrend_array(arr, axis=1):
    return signal.detrend(arr, axis=axis)


def minmax_normalize_array(arr, axis=None):
    out = arr - arr.min(axis=axis, keepdims=True)
    out = out / out.max(axis=axis, keepdims=True)
    out = np.where(np.isfinite(out), out, np.zeros_like(out))

    return out


def get_nii(path, fwhm=None, add_to_cache=True, nii_cache=NII_CACHE):
    if path not in nii_cache:
        img = image.smooth_img(path, fwhm)
        if add_to_cache:
            nii_cache[path] = img
    else:
        img = nii_cache[path]
    return img


def get_atlas(atlas, fwhm=None):
    if isinstance(atlas, str) and atlas.lower() == 'language':
        name = 'language'
        filename = 'LanA_n806.nii'
        with pkg_resources.as_file(pkg_resources.files(resources).joinpath(filename)) as path:
            val = get_nii(path, fwhm=fwhm)
    elif isinstance(atlas, dict):
        keys = list(atlas.keys())
        assert len(keys) == 1, 'If reference_network is provided as a dict, must contain exactly one entry. ' + \
                               'Got %d entries for reference network %s.' % (len(keys), atlas)
        name = keys[0]
        val = atlas[name]
        if isinstance(val, str):
            path = val
        else:
            path = None
    else:
        assert len(atlas) == 2, 'Reference network must be a pair: <path, value>.'
        name, val = atlas
        if isinstance(val, str):
            path = val
        else:
            path = None
    if isinstance(val, str):
        val = get_nii(val, fwhm=fwhm)
    assert 'Nifti1Image' in type(val).__name__, \
        'Atlas must be either a string path or a Nifti-like image class. Got type %s.' % type(val)

    return name, path, val


def get_shape_from_parcellations(parcellations):
    n_samples = parcellations.shape[0]
    v = parcellations.shape[1]
    if len(parcellations.shape) == 2:
        n_networks = parcellations.max() + 1
    else:
        n_networks = parcellations.shape[2]

    return n_samples, v, n_networks

def align_samples(samples, ref_ix, scoring_method='corr', w=None):
    scoring_method = scoring_method.lower()
    n_samples, v, n_networks = get_shape_from_parcellations(samples)
    parcellation = np.zeros((n_networks, v))
    reference = samples[ref_ix]
    if len(samples.shape) == 2:
        reference = (reference[None, ...] == np.arange(n_networks)[..., None]).astype(float)
    else:
        reference = reference.T.astype(float)
    if scoring_method == 'corr':
        _reference = standardize_array(reference)
    else:
        _reference = reference

    # Align subsequent samples
    for si in range(n_samples):
        if w is None or w[si]:
            if len(samples.shape) == 2:
                s = (samples[si][None, ...] == np.arange(n_networks)[..., None]).astype(float)
            else:
                s = samples[si].T.astype(float)
            if si != ref_ix:
                if scoring_method == 'corr':
                    s = standardize_array(s)
                    scores = np.dot(
                        s,
                        _reference.T
                    ) / v
                elif scoring_method == 'avg':
                    num = np.dot(
                        s,
                        _reference.T
                    )
                    denom = s.sum(axis=-1, keepdims=True)
                    denom[np.where(denom == 0)] = 1
                    scores = num / denom
                else:
                    raise ValueError('Unrecognized scoring method %s.' % scoring_method)

                ix_l, ix_r = optimize.linear_sum_assignment(scores, maximize=True)
                # Make sure networks are sorted in the same order as current parcellation
                sort_ix = np.argsort(ix_l)
                ix_l, ix_r = ix_l[sort_ix], ix_r[sort_ix]
                s = s[ix_r]
                if w is not None:
                    s = s * w[si]
            parcellation = parcellation + s
    if w is not None:
        denom = w.sum()
    else:
        denom = n_samples
    parcellation = parcellation / denom

    return parcellation


def purge_bad_nii(path, compressed=True):
    if os.path.exists(path):
        suffix = get_suffix(compressed)
        for sub in os.listdir(path):
            _path = os.path.join(path, sub)
            if os.path.isdir(_path):
                purge_bad_nii(_path, compressed=compressed)
            elif _path.endswith(suffix):
                try:
                    image.load_img(_path)
                except Exception as e:
                    stderr('Removing corrupted file: % s\n' % _path)
                    stderr('Exception:\n')
                    stderr(textwrap.indent(str(e), '    '))
                    os.remove(_path)


class Data:
    def __new__(cls, *args, **kwargs):
        if cls is Data:
            raise TypeError(f"{cls.__name__} is an abstract class that cannot be instantiated")
        return super().__new__(cls)

    def __init__(
            self,
            nii_ref_path,
            fwhm=None,
    ):
        self.nii_ref_path = nii_ref_path
        self.fwhm = fwhm
        self.nii_ref = get_nii(self.nii_ref_path, fwhm=self.fwhm)
        self.nii_ref_shape = self.nii_ref.shape[:3]
        self.mask = None
        self.set_mask_from_nii(None)

    @property
    def v(self):
        return self.mask.sum()

    def set_mask_from_nii(self, mask_path):
        if mask_path is None:
            mask = masking.compute_brain_mask(self.nii_ref, connected=False, opening=False, mask_type='gm')
        else:
            mask = get_nii(mask_path, fwhm=self.fwhm)
        mask = image.get_data(mask) > 0.5
        self.mask = mask

    def get_bandpass_filter(self, tr, lower=None, upper=None, order=5):
        assert lower is not None or upper is not None, 'At least one of the lower (hi-pass) or upper (lo-pass) ' + \
                                                       'parameters must be provided.'
        fs = 1/tr
        Wn = []
        btype = None
        if lower is not None:
            Wn.append(lower)
            btype = 'highpass'
        if upper is not None:
            Wn.append(upper)
            if btype is None:
                btype = 'lowpass'
            else:
                btype = 'bandpass'
        if len(Wn) == 1:
            Wn = Wn[0]

        return signal.butter(order, Wn, fs=fs, btype=btype)

    def bandpass(self, arr, lower=None, upper=None, order=5, axis=-1):
        if lower is None and upper is None:
            return arr
        b, a = self.get_bandpass_filter(lower, upper, order=order)
        out = signal.lfilter(b, a, arr, axis=axis)

        return out

    def flatten(self, nii):
        arr = image.get_data(nii)
        arr = arr[self.mask]

        return arr

    def unflatten(self, arr):
        shape = tuple(self.nii_ref_shape) + tuple(arr.shape[1:])
        out = np.zeros(shape, dtype=arr.dtype)
        out[self.mask] = arr
        nii = image.new_img_like(self.nii_ref, out)

        return nii


class InputData(Data):
    def __init__(
            self,
            functional_paths,
            fwhm=None,
            mask_path=None,
            standardize=True,
            normalize=False,
            detrend = False,
            tr=2,
            low_pass=None,
            high_pass=None,
            compress_outputs=True
    ):
        if isinstance(functional_paths, str):
            functional_paths = [functional_paths]
        else:
            functional_paths = list(functional_paths)
        assert len(functional_paths), 'At least one functional run must be provided for parcellation.'
        nii_ref_path = functional_paths[0]
        super().__init__(nii_ref_path, fwhm=fwhm)

        # Load all data and aggregate the mask
        _functionals = []
        _mask = None
        for functional_path in functional_paths:
            functional = get_nii(functional_path, fwhm=self.fwhm)
            data = image.get_data(functional)
            __mask = (data.std(axis=-1) > 0) & \
                     np.all(np.isfinite(data), axis=-1)  # Mask all voxels with NaNs or with no variance
            if __mask.sum() == 0:
                stderr('No valid voxels (finite-valued, sd > 0) found in image %s. Skipping.\n' % functional_path)
                continue
            if _mask is None:
                _mask = __mask
            else:
                _mask &= __mask
            _functionals.append(functional)
        functionals = _functionals

        self.set_mask_from_nii(mask_path)
        self.mask = self.mask & _mask
        mask = self.mask

        # Set key variables now so they can be used in instance methods during initialization
        self.tr = tr
        self.low_pass = low_pass
        self.high_pass = high_pass

        # Perform any post-processing
        for i, functional in enumerate(functionals):
            functional = self.flatten(functional)
            functional = self.bandpass(functional)  # self.bandpass() is a no-op if no bandpassing parameters are set
            if standardize:
                functional = standardize_array(functional)
            if detrend:
                functional = detrend_array(functional)
            if normalize:
                functional = sk_normalize(functional, axis=1)
            functionals[i] = functional

        self.nii_ref = nii_ref_path
        self.functionals = functionals
        self.timecourses = np.concatenate(self.functionals, axis=-1)
        self.compress_outputs = compress_outputs

    @property
    def n_trs(self):
        return self.timecourses.shape[-1]

    @property
    def n_runs(self):
        return len(self.functionals)

    def get_bandpass_filter(self, tr=None, lower=None, upper=None, order=5):
        assert lower is not None or upper is not None, 'At least one of the lower (hi-pass) or upper (lo-pass) ' + \
                                                       'parameters must be provided.'
        if tr is None:
            tr = self.tr

        return super().get_bandpass_filter(tr=tr, lower=lower, upper=upper, order=order)

    def save_atlases(self, output_dir, compress_outputs=None):
        if compress_outputs is None:
            compress_outputs = self.compress_outputs
        suffix = get_suffix(compress_outputs)
        mask = image.new_img_like(self.nii_ref, self.mask)
        mask.to_filename(join(output_dir, 'mask%s' % suffix))


class ReferenceData(Data):
    def __init__(
            self,
            reference_atlases=None,
            fwhm=None,
            compress_outputs=True
    ):

        if reference_atlases is None:
            reference_atlases = []
        elif isinstance(reference_atlases, str):
            reference_atlases = []

        assert len(reference_atlases), 'At least one reference atlas must be provided for evaluation.'

        # Load
        _reference_atlases = {}  # Structure: atlas_name: atlas_nii
        reference_atlas_names = []
        nii_ref_path = None
        for i, reference_atlas in enumerate(reference_atlases):
            if isinstance(reference_atlases, dict):
                reference_atlas = {reference_atlas: reference_atlases[reference_atlas]}
            reference_atlas, reference_atlas_path, val = get_atlas(reference_atlas, fwhm=fwhm)
            if nii_ref_path is None:
                nii_ref_path = reference_atlas_path
            reference_atlas_names.append(reference_atlas)
            _reference_atlases[reference_atlas] = val
        reference_atlases = _reference_atlases

        super().__init__(nii_ref_path, fwhm=fwhm)

        # Perform any post-processing and save reference/evaluation images
        for key in reference_atlases:
            val = reference_atlases[key]
            val = self.flatten(val)
            # val = standardize_array(val)
            reference_atlases[key] = val

        self.reference_atlases = reference_atlases
        self.reference_atlas_names = reference_atlas_names
        self.compress_outputs = compress_outputs

    def save_atlases(self, output_dir, compress_outputs=None):
        if compress_outputs is None:
            compress_outputs = self.compress_outputs
        suffix = get_suffix(compress_outputs)

        # Perform any post-processing and save reference/evaluation images
        reference_atlases = self.reference_atlases
        for key in reference_atlases:
            val = self.unflatten(reference_atlases[key])
            val.to_filename(join(output_dir, '%s%s%s' % (REFERENCE_ATLAS_PREFIX, key, suffix)))


class EvaluationData(Data):
    def __init__(
            self,
            evaluation_atlases=None,
            fwhm=None,
            compress_outputs=True
    ):
        if evaluation_atlases is None:
            evaluation_atlases = {}

        assert len(evaluation_atlases), 'At least one evaluation atlas must be provided for evaluation.'

        # Load
        _evaluation_atlases = {}  # Structure: reference_atlas_name: evaluation_atlas_name: atlas_nii
        nii_ref_path = None
        for i, reference_atlas in enumerate(evaluation_atlases):
            for evaluation_atlas in evaluation_atlases[reference_atlas]:
                if isinstance(evaluation_atlases[reference_atlas], dict):
                    evaluation_atlas = {evaluation_atlas: evaluation_atlases[reference_atlas][evaluation_atlas]}
                evaluation_atlas, evaluation_atlas_path, val = get_atlas(evaluation_atlas, fwhm=fwhm)
                if nii_ref_path is None:
                    nii_ref_path = evaluation_atlas_path
                if reference_atlas not in _evaluation_atlases:
                    _evaluation_atlases[reference_atlas] = {}
                _evaluation_atlases[reference_atlas][evaluation_atlas] = val
        evaluation_atlases = _evaluation_atlases

        super().__init__(nii_ref_path, fwhm=fwhm)

        # Perform any post-processing and save evaluation images
        for reference_atlas in evaluation_atlases:
            _evaluation_atlases = evaluation_atlases[reference_atlas]
            for key in _evaluation_atlases:
                val = _evaluation_atlases[key]
                val = self.flatten(val)
                # val = standardize_array(val)
                _evaluation_atlases[key] = val

        self.evaluation_atlases = evaluation_atlases
        self.compress_outputs = compress_outputs

    def save_atlases(self, output_dir, compress_outputs=None):
        if compress_outputs is None:
            compress_outputs = self.compress_outputs
        suffix = get_suffix(compress_outputs)

        # Perform any post-processing and save reference/evaluation images
        evaluation_atlases = self.evaluation_atlases
        for reference_atlas in evaluation_atlases:
            _evaluation_atlases = evaluation_atlases[reference_atlas]
            for key in _evaluation_atlases:
                val = self.unflatten(_evaluation_atlases[key])
                val.to_filename(join(output_dir, '%s%s_%s%s' % (EVALUATION_ATLAS_PREFIX, reference_atlas, key, suffix)))
