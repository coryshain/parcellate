import os
import textwrap

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from parcellate import resources
import numpy as np
from scipy import signal, optimize
from nilearn import image, masking

from parcellate.util import REFERENCE_ATLAS_PREFIX, EVALUATION_ATLAS_PREFIX, ALL_REFERENCE, join, get_suffix, stderr

NII_CACHE = {}  # Map from paths to NII objects
ATLAS_NAME_TO_FILE = dict(
    language='LanA_n806.nii',
    lana='LanA_n806.nii',
    executive='fROI20_HE197.nii',
    md='fROI20_HE197.nii',
    aud='DU15_AUD.nii.gz',
    cg_op='DU15_CG_OP.nii.gz',
    datn_a='DU15_dATN_A.nii.gz',
    datn_b='DU15_dATN_B.nii.gz',
    dn_a='DU15_DN_A.nii.gz',
    dn_b='DU15_DN_B.nii.gz',
    fpn_a='DU15_FPN_A.nii.gz',
    fpn_b='DU15_FPN_B.nii.gz',
    lang='DU15_LANG.nii.gz',
    pm_ppr='DU15_PM_PPr.nii.gz',
    sal_pmn='DU15_SAL_PMN.nii.gz',
    smot_a='DU15_SMOT_A.nii.gz',
    smot_b='DU15_SMOT_B.nii.gz',
    vis_c='DU15_VIS_C.nii.gz',
    vis_p='DU15_VIS_P.nii.gz',
)


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


def get_nii(path, fwhm=None, add_to_cache=True, nii_cache=NII_CACHE, threshold=None):
    if path not in nii_cache:
        img = image.smooth_img(path, fwhm)
        if add_to_cache:
            nii_cache[path] = img
    else:
        img = nii_cache[path]
    if threshold is not None:
        data = image.get_data(img)
        data = binarize_array(data, threshold=threshold)
        img = image.new_img_like(img, data)
    return img


def get_atlas(atlas, fwhm=None, threshold=None):
    if isinstance(atlas, str):
        name = atlas
        filename = ATLAS_NAME_TO_FILE.get(name.lower(), None)
        if filename is None:
            raise ValueError('Unrecognized atlas name: %s' % name)
        with pkg_resources.as_file(pkg_resources.files(resources).joinpath(filename)) as path:
            val = get_nii(path, fwhm=fwhm, threshold=threshold)
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
        val = get_nii(val, fwhm=fwhm, threshold=threshold)
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


def align_samples(
        samples,
        scoring_method='corr',
        w=None,
        n_alignments=None,
        shuffle=False,
        greedy=True,
        indent=None
):
    if w is None:
        _w = 1
    else:
        _w = w[0]
    scoring_method = scoring_method.lower()
    n_samples, v, n_networks = get_shape_from_parcellations(samples)
    reference = (samples[0][None, ...] == np.arange(n_networks)[..., None]).astype(float)
    parcellation = None
    C = 0

    # Align subsequent samples
    if shuffle:
        s_ix = np.random.permutation(n_samples)
        samples = samples[s_ix]
    n = n_alignments
    if n is None:
        n = n_samples
    i = 0
    for i_cum in range(n):
        if indent is not None:
            stderr('\r%sAlignment %d/%d' % (' ' * (indent * 2), i_cum + 1, n))

        if w is not None:
            _w = w[i]
        else:
            _w = 1
        if _w == 0:
            continue

        if len(samples.shape) == 2:
            s = (samples[i][None, ...] == np.arange(n_networks)[..., None])
        else:
            s = samples[i].T
        s = s.astype(float)
        if scoring_method == 'corr':
            _reference = standardize_array(reference)
            _s = standardize_array(s)
            scores = np.dot(
                _reference,
                _s.T,
            ) / v
        elif scoring_method == 'avg':
            _reference = minmax_normalize_array(reference)
            num = np.dot(
                _reference,
                s.T
            )
            denom = s.sum(axis=-1, keepdims=True)
            denom[np.where(denom == 0)] = 1
            scores = num / denom
        else:
            raise ValueError('Unrecognized scoring method %s.' % scoring_method)

        _, ix_r = optimize.linear_sum_assignment(scores, maximize=True)
        s = s[ix_r]
        if parcellation is None:
            parcellation = s * _w
        else:
            parcellation = parcellation + s * _w
        if greedy:
            reference = parcellation
        C += _w
        i += 1
        if i >= n_samples:
            i = 0
            if shuffle:
                s_ix = np.random.permutation(n_samples)
                samples = samples[s_ix]

    parcellation = parcellation / C

    stderr('\n')

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

def resample_to(nii, template):
    nii = image.math_img('nii * (1 + 1e-6)', nii=nii)  # Hack to force conversion to float
    return image.resample_to_img(nii, template)


class Data:
    def __init__(
            self,
            nii_ref_path,
            fwhm=None,
            resampling_target_nii=None
    ):
        self.nii_ref_path = nii_ref_path
        self.fwhm = fwhm
        self.nii_ref = get_nii(self.nii_ref_path, fwhm=self.fwhm)
        if resampling_target_nii is not None:
            self.nii_ref = resample_to(self.nii_ref, resampling_target_nii)
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

    def get_bandpass_filter(self, tr=None, lower=None, upper=None, order=5):
        assert lower is not None or upper is not None, 'At least one of the lower (hi-pass) or upper (lo-pass) ' + \
                                                       'parameters must be provided.'
        assert tr is not None, 'TR must be provided.'
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

    def bandpass(self, arr, tr=None, lower=None, upper=None, order=5, axis=-1):
        if (lower is None and upper is None) or tr is None:
            return arr
        b, a = self.get_bandpass_filter(tr=tr, lower=lower, upper=upper, order=order)
        out = signal.lfilter(b, a, arr, axis=axis)

        return out

    def flatten(self, nii):
        arr = image.get_data(nii)
        arr = arr[self.mask]

        return arr

    def unflatten(self, arr, mask=None, nii_ref=None):
        if mask is None:
            mask = self.mask
        if nii_ref is None:
            nii_ref = self.nii_ref
            nii_ref_shape = self.nii_ref_shape
        else:
            nii_ref_shape = image.get_data(nii_ref).shape[:3]
        shape = tuple(nii_ref_shape) + tuple(arr.shape[1:])
        out = np.zeros(shape, dtype=arr.dtype)
        out[mask] = arr
        nii = image.new_img_like(nii_ref, out)

        return nii


class InputData(Data):
    def __init__(
            self,
            functional_paths,
            fwhm=None,
            resampling_target_nii=None,
            mask_path=None,
            detrend=False,
            standardize=True,
            envelope=False,
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
        if isinstance(resampling_target_nii, str):
            resampling_target_nii = image.smooth_img(resampling_target_nii, None)
        super().__init__(nii_ref_path, fwhm=fwhm, resampling_target_nii=resampling_target_nii)

        # Load all data and aggregate the mask
        _functionals = []
        _mask = None
        for functional_path in functional_paths:
            try:
                functional = get_nii(functional_path, fwhm=self.fwhm)
            except OSError as e:
                stderr('Error loading image %s: %s.\n\n File may be corrupted. Skipping.\n' % (functional_path, e))
                continue
            if resampling_target_nii is not None:
                functional = resample_to(functional, resampling_target_nii)
            data = image.get_data(functional)
            if len(data.shape) > 3:
                __mask = (data.std(axis=-1) > 0) & \
                         np.all(np.isfinite(data), axis=-1)  # Mask all voxels with NaNs or with no variance
            else:  # Not a timeseries, or a single TR, don't reduce along time axis
                __mask = np.isfinite(data)
                functional = image.new_img_like(functional, data[..., None])
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

        # Set key variables now so they can be used in instance methods during initialization
        self.tr = tr
        self.low_pass = low_pass
        self.high_pass = high_pass

        # Perform any post-processing
        for i, functional in enumerate(functionals):
            functional = self.flatten(functional)
            if envelope:
                functional = np.abs(signal.hilbert(functional, axis=-1))
            # self.bandpass() is a no-op if no bandpassing parameters are set
            functional = self.bandpass(functional, tr=self.tr, lower=self.high_pass, upper=self.low_pass)
            if detrend:
                functional = detrend_array(functional)
            if standardize:
                functional = standardize_array(functional)
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


class AtlasData(Data):
    def __init__(
            self,
            atlases=None,
            fwhm=None,
            network_threshold=None,
            resampling_target_nii=None,
            compress_outputs=True
    ):

        if atlases is None:
            atlases = []
        elif isinstance(atlases, str):
            if isinstance(atlases, str) and atlases.lower() in ('default', 'all', 'all_reference'):
                atlases = ALL_REFERENCE
            else:
                atlases = [atlases]
        elif isinstance(atlases, list):
            _atlases = []
            for atlas in atlases:
                if isinstance(atlas, str) and atlas.lower() in ('default', 'all', 'all_reference'):
                    atlas = ALL_REFERENCE
                else:
                    atlas = [atlas]
                _atlases.extend(atlas)
            atlases = _atlases

        if len(atlases) == 0:
            with pkg_resources.as_file(pkg_resources.files(resources).joinpath(ATLAS_NAME_TO_FILE['lang'])) as path:
                resampling_target_nii = path
        resampling_target_nii_path = resampling_target_nii

        # Load
        if isinstance(resampling_target_nii, str):
            resampling_target_nii = image.smooth_img(resampling_target_nii, None)
        _atlases = {}  # Structure: atlas_name: atlas_nii
        _atlas_names = []
        nii_ref_path = None
        for i, reference_atlas in enumerate(atlases):
            if isinstance(reference_atlas, str) and isinstance(atlases, dict):
                reference_atlas = {reference_atlas: atlases[reference_atlas]}
            reference_atlas, reference_atlas_path, nii = get_atlas(
                reference_atlas, fwhm=fwhm, threshold=network_threshold
            )
            if resampling_target_nii is not None:
                nii = resample_to(nii, resampling_target_nii)
            if nii_ref_path is None:
                nii_ref_path = reference_atlas_path
            _atlas_names.append(reference_atlas)
            _atlases[reference_atlas] = nii
        if nii_ref_path is None:
            nii_ref_path = resampling_target_nii_path
        atlases = _atlases

        super().__init__(nii_ref_path, fwhm=fwhm, resampling_target_nii=resampling_target_nii)

        # Perform any post-processing and save reference/evaluation images
        for key in atlases:
            nii = atlases[key]
            nii = self.flatten(nii)
            atlases[key] = nii

        self.atlases = atlases
        self.atlas_names = _atlas_names
        self.compress_outputs = compress_outputs

    def save_atlases(self, output_dir, prefix='', compress_outputs=None):
        if compress_outputs is None:
            compress_outputs = self.compress_outputs
        suffix = get_suffix(compress_outputs)

        # Perform any post-processing and save reference/evaluation images
        reference_atlases = self.atlases
        for key in reference_atlases:
            val = self.unflatten(reference_atlases[key])
            val.to_filename(join(output_dir, '%s%s%s' % (prefix, key, suffix)))
