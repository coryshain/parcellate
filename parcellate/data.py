import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
import resources
import numpy as np
from scipy import signal
from sklearn.preprocessing import normalize as sk_normalize
from nilearn import image, masking, plotting


def get_atlas(atlas):
    if isinstance(atlas, str) and atlas.lower() == 'language':
        key = atlas
        filename = 'LanA_n806.nii'
        with pkg_resources.as_file(pkg_resources.files(resources).joinpath(filename)) as path:
            val = image.smooth_img(path, None)
    elif isinstance(atlas, dict):
        keys = list(atlas.keys())
        assert len(keys) == 1, 'If reference_network is provided as a dict, must contain exactly one entry. ' + \
                               'Got %d entries for reference network %s.' % (len(keys), atlas)
        key = keys[0]
        val = atlas[key]
    else:
        assert len(atlas) == 2, 'Reference network must be a pair: <name, value>.'
        atlas, val = atlas
    if isinstance(val, str):
        val = image.load_img(val)
    assert 'Nifti1Image' in type(val).__name__, \
        'Atlas must be either a string path or a Nifti-like image class. Got type %s.' % type(val)

    return key, val

class ParcellateData:
    def __init__(
            self,
            functionals,
            mask=None,
            standardize=True,
            normalize=False,
            detrend = False,
            tr=2,
            low_pass=None,
            high_pass=None,
            reference_atlases=None,
            evaluation_atlases=None,
            atlas_lower_cutoff=None,
            atlas_upper_cutoff=None,
    ):
        if isinstance(functionals, str) or 'Nifti1Image' in type(functionals).__name__:
            functionals = [functionals]
        else:
            try:
                functionals = list(functionals)
            except TypeError:
                raise ValueError('functionals must be a string, a Nifti image, or a list-like of the above. Got %s.' \
                                 % type(functionals).__name__)
        assert len(functionals), 'At least one functional run must be provided for parcellation.'

        if reference_atlases is None:
            reference_atlases = ['language']
        if evaluation_atlases is None:
            evaluation_atlases = []

        # Load all data and aggregate the mask
        _functionals = []
        nii_ref_tmp = None
        _mask = None
        for functional in functionals:
            if isinstance(functional, str):
                functional = image.smooth_img(functional, None)
            assert 'Nifti1Image' in type(
                functional).__name__, 'Functional must be either a string path or a Nifti-like' + \
                                      'image class. Got type %s.' % type(functional)
            data = image.get_data(functional)
            m, s = data.mean(axis=-1), data.std(axis=-1)
            if standardize:
                data = (data - m[..., None]) / s[..., None]
            functional = image.new_img_like(functional, data)
            __mask = (s > 0) & np.all(np.isfinite(data), axis=-1)  # Mask all voxels with NaNs or with no variance
            if nii_ref_tmp is None:
                nii_ref_tmp = functional
            if _mask is None:
                _mask = __mask
            else:
                _mask &= __mask
            _functionals.append(functional)
        functionals = _functionals

        _reference_atlases = {}  # Structure: atlas_name: atlas_nii
        reference_atlas_names = []
        for i, reference_atlas in enumerate(reference_atlases):
            if isinstance(reference_atlases, dict):
                reference_atlas = {reference_atlas: reference_atlases[reference_atlas]}
            key, val = get_atlas(reference_atlas)
            reference_atlas_names.append(key)
            _reference_atlases[key] = val
            reference_atlas = image.get_data(val)
            _mask &= np.isfinite(reference_atlas)
        reference_atlases = _reference_atlases

        _evaluation_atlases = {}  # Structure: reference_atlas_name: evaluation_atlas_name: atlas_nii
        for i, reference_atlas in enumerate(evaluation_atlases):
            if reference_atlas in reference_atlases:
                for evaluation_atlas in evaluation_atlases[reference_atlas]:
                    if isinstance(evaluation_atlases[reference_atlas], dict):
                        evaluation_atlas = {evaluation_atlas: evaluation_atlases[reference_atlas][evaluation_atlas]}
                    key, val = get_atlas(evaluation_atlas)
                    if reference_atlas not in _evaluation_atlases:
                        _evaluation_atlases[reference_atlas] = {}
                    _evaluation_atlases[reference_atlas][key] = val
        evaluation_atlases = _evaluation_atlases

        if mask is None:
            mask = masking.compute_brain_mask(nii_ref_tmp, connected=False, opening=False, mask_type='gm')
            nii_ref = mask
        else:
            mask = image.smooth_img(mask, None)
            nii_ref = mask
            mask = image.math_img('img > 0.5', img=mask)
        mask = image.get_data(mask)
        mask = (mask > 0.5) & _mask

        # Set key variables now so they can be used in instance methods during initialization
        self.mask = mask
        self.tr = tr
        self.low_pass = low_pass
        self.high_pass = high_pass

        nii_ref_shape = mask.shape

        v = int(mask.sum())

        # Perform any post-processing and save reference/evaluation images
        for key in reference_atlases:
            val = reference_atlases[key]
            val = self.flatten(val)
            if atlas_lower_cutoff is not None or atlas_upper_cutoff is not None:
                val = np.clip(val, atlas_lower_cutoff, atlas_upper_cutoff)
            val = self.standardize(val)
            reference_atlases[key] = val

        for reference_atlas in evaluation_atlases:
            _evaluation_atlases = evaluation_atlases[reference_atlas]
            for key in _evaluation_atlases:
                val = _evaluation_atlases[key]
                val = self.flatten(val)
                if atlas_lower_cutoff is not None or atlas_upper_cutoff is not None:
                    val = np.clip(val, atlas_lower_cutoff, atlas_upper_cutoff)
                val = self.standardize(val)
                _evaluation_atlases[key] = val

        for i, functional in enumerate(functionals):
            functional = self.flatten(functional)
            if detrend:
                functional = self.detrend(functional)
            if normalize:
                functional = sk_normalize(functional, axis=1)
            functional = self.bandpass(functional)  # self.bandpass() is a no-op if no bandpassing parameters are set
            functionals[i] = functional

        self.nii_ref = nii_ref
        self.nii_ref_shape = nii_ref_shape
        self.functionals = functionals
        self.evaluation_atlases = evaluation_atlases
        self.reference_atlases = reference_atlases
        self.reference_atlas_names = reference_atlas_names
        self.v = v
        self.timecourses = np.concatenate(self.functionals, axis=-1)

    def standardize(self, arr):
        return (arr - arr.mean(axis=-1, keepdims=True)) / arr.std(axis=-1, keepdims=True)

    def detrend(self, arr):
        return signal.detrend(arr, axis=-1)

    def get_bandpass_filter(self, lower=None, upper=None, order=5):
        assert lower is not None or upper is not None, 'At least one of the lower (hi-pass) or upper (lo-pass) ' + \
                                                       'parameters must be provided.'
        fs = 1/self.tr
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

    def bandpass(self, arr, lower=None, upper=None, order=5):
        if lower is None:
            lower = self.high_pass
        if upper is None:
            upper = self.low_pass
        if lower is None and upper is None:
            return arr
        b, a = self.get_bandpass_filter(lower, upper, order=order)
        out = signal.lfilter(b, a, arr, axis=-1)
        return out

    def minmax_normalize(self, arr, eps=1e-8):
        arr -= arr.min()
        arr /= arr.max() + eps
        return arr

    def flatten(self, arr):
        arr = image.get_data(arr)
        return arr[self.mask]

    def unflatten(self, arr):
        out = np.zeros(self.nii_ref_shape, dtype='float32')
        out[self.mask] = arr
        out = image.new_img_like(self.nii_ref, out)

        return out

    def save_atlases(self, output_dir):
        mask = image.new_img_like(self.nii_ref, self.mask)
        mask.to_filename(os.path.join(output_dir, 'mask.nii'))

        # Perform any post-processing and save reference/evaluation images
        reference_atlases = self.reference_atlases
        for key in reference_atlases:
            val = self.unflatten(reference_atlases[key])
            val.to_filename(os.path.join(output_dir, 'reference_atlas_%s.nii' % key))

        evaluation_atlases = self.evaluation_atlases
        for reference_atlas in evaluation_atlases:
            _evaluation_atlases = evaluation_atlases[reference_atlas]
            for key in _evaluation_atlases:
                val = self.unflatten(_evaluation_atlases[key])
                val.to_filename(os.path.join(output_dir, 'evaluation_atlas_%s_%s.nii' % (reference_atlas, key)))
