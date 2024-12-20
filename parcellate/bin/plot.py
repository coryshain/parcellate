import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
from nilearn import image
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from PIL import Image
import textwrap
import pprint
import argparse

from parcellate.cfg import *
from parcellate.util import *


######################################
#
#  GET BETTER FONT
#
######################################

roboto_url = 'https://github.com/google/fonts/blob/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf'
url = roboto_url + '?raw=true'
response = urlopen(url)
f = NamedTemporaryFile(suffix='.ttf')
f.write(response.read())
fm.fontManager.addfont(f.name)
prop = fm.FontProperties(fname=f.name)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


######################################
#
#  CONSTANTS
#
######################################

SUFFIX2NAME = {
    '_atpgt0.1': 'p > 0.1',
    '_atpgt0.2': 'p > 0.2',
    '_atpgt0.3': 'p > 0.3',
    '_atpgt0.4': 'p > 0.4',
    '_atpgt0.5': 'p > 0.5',
    '_atpgt0.6': 'p > 0.6',
    '_atpgt0.7': 'p > 0.7',
    '_atpgt0.8': 'p > 0.8',
    '_atpgt0.9': 'p > 0.9',
}

COLORS = np.array([
    [255, 0, 0],      # Red
    [0, 0, 255],      # Blue
    [0, 255, 0],      # Green
    [0, 255, 255],    # Cyan
    [255, 0, 255],    # Magenta
    [255, 255, 0],    # Yellow
    [255, 128, 0],    # Orange
    [255, 0, 128],    # Pink
    [128, 255, 0],    # Lime
    [0, 128, 255],    # Aqua
    [128, 0, 255],    # Violet
    [0, 255, 128],    # Teal
    [255, 64, 0],     # Fire
    [255, 0, 64],     # Hot Pink
    [0, 64, 255],     # Ocean
], dtype=int)
RED = COLORS[0]
BLUE = COLORS[1]
GREEN = COLORS[2]
BASE_BRIGHTNESS = 0.  # Value from 0. (black) to 1. (full color)








######################################
#
#  UTILITIES
#
######################################

def sample_color():
    r, g, b = np.random.random(size=3)
    max_val = max(r, g, b)
    r = round(r / max_val * 255)
    g = round(g / max_val * 255)
    b = round(b / max_val * 255)

    return r, g, b

def expand_color(color, base_brightness=BASE_BRIGHTNESS):
    out = tuple([
        int(round(x * base_brightness)) for x in color
    ]) + tuple(color)

    return out







######################################
#
#  ATLAS
#
######################################


def plot_atlases(
        cfg_paths,
        parcellation_ids=None,
        subnetwork_id=1,
        atlas_names=None,
        atlas_types=None,
        dark_atlas=False,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
        clip_p=None,
        clip_act=None,
        overwrite_atlases=False
):
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]

    binary_dir = join(dirname(dirname(__file__)), 'resources', 'surfice', 'Surf_Ice')
    assert os.path.exists(binary_dir), ('Surf Ice directory %s not found. Install using '
        '``python -m parcellate.bin.install_surf_ice``.' % binary_dir)
    binary_path = None
    for path in os.listdir(binary_dir):
        if path in ('surfice', 'surfice.exe'):
            binary_path = join(binary_dir, path)
            break
    assert binary_path, 'No Surf Ice executable found'

    with TemporaryDirectory() as tmp_dir_path:
        for cfg_path in cfg_paths:
            if not os.path.exists(cfg_path):
                continue
            print(cfg_path)
            cfg = get_cfg(cfg_path)
            output_dir = cfg['output_dir']

            parcellation_path = join(output_dir, 'parcellation')
            if not os.path.exists(parcellation_path):
                continue

            if overwrite_atlases:
                skip = False
            else:
                skip = True
                for parcellation_dir in os.listdir(parcellation_path):
                    if parcellation_ids is None or \
                            parcellation_dir in parcellation_ids or \
                            parcellation_dir == parcellation_ids:
                        parcellation_dir = join(output_dir, 'parcellation', parcellation_dir, 'plots')
                        breadcrumb_path = os.path.join(parcellation_dir, 'finished.txt')
                        if not os.path.exists(breadcrumb_path):
                            skip = False
                            break

            if skip:
                continue

            atlas_paths = _get_atlas_paths(
                cfg_path,
                parcellation_ids=parcellation_ids,
                reference_atlas_names=reference_atlas_names,
                evaluation_atlas_names=evaluation_atlas_names
            )

            atlas_paths_orig = update_atlas_paths(atlas_paths, tmp_dir_path)
            if not len(atlas_paths_orig):
                continue
            copy_paths = list(atlas_paths_orig) + [tmp_dir_path]

            print('  Copying atlases to temporary directory...')

            subprocess.call(['cp'] + copy_paths)

            min_by_path = {}
            max_by_path = {}
            for x in os.listdir(tmp_dir_path):
                if not x.endswith('.nii.gz'):
                    continue
                path = os.path.join(tmp_dir_path, x)
                img = image.load_img(path)
                data = image.get_data(img)
                thresh = None
                if 'eval' in x:
                    if clip_act is not None:
                        thresh = clip_act
                elif clip_p is not None:
                    thresh = clip_p
                if thresh is not None:
                    data = np.where(np.logical_and(np.isfinite(data), data < thresh), 0, data)
                    img = image.new_img_like(img, data)
                    img.to_filename(path)
                data = image.get_data(img)
                min_by_path[path] = np.nanmin(data)
                max_by_path[path] = np.nanmax(data)

            script = _get_surf_ice_script(
                [cfg_path],
                atlas_paths=atlas_paths,
                atlas_names=atlas_names,
                atlas_types=atlas_types,
                dark_atlas=dark_atlas,
                subnetwork_id=subnetwork_id
            )

            tmp_path = os.path.join(tmp_dir_path, 'PARCELLATE_SURFICE_SCRIPT_TMP.py')
            with open(tmp_path, 'w') as f:
                f.write(script)

            print('  Generating subplots...')

            subprocess.call([binary_path, '-S', tmp_path])

            print('  Stitching plots...')
            if os.path.exists(join(tmp_dir_path, 'parcellation')):
                parcellation_dirs = os.listdir(join(tmp_dir_path, 'parcellation'))
            else:
                parcellation_dirs = []
            for parcellation_dir in parcellation_dirs:
                if parcellation_ids is None or \
                        parcellation_dir in parcellation_ids or \
                        parcellation_dir == parcellation_ids:
                    parcellation_id = parcellation_dir
                    parcellation_dir = join(tmp_dir_path, 'parcellation', parcellation_id, 'plots')
                    img_prefixes = set()
                    for img in [x for x in os.listdir(parcellation_dir) if _is_hemi(x)]:
                        img_prefix = '_'.join(img.split('_')[:-2])
                        img_prefix = join(parcellation_dir, img_prefix)
                        img_prefixes.add(img_prefix)
                    for img_prefix in img_prefixes:
                        imgs = []
                        img_paths = []
                        for hemi in ('left', 'right'):
                            if hemi == 'left':
                                views = ('lateral', 'medial')
                            else:
                                views = ('medial', 'lateral')
                            for view in views:
                                img_path = img_prefix + '_%s_%s.png' % (hemi, view)
                                imgs.append(Image.open(img_path))
                                img_paths.append(img_path)
                        widths, heights = zip(*(i.size for i in imgs))
                        total_width = sum(widths)
                        max_height = max(heights)
                        new_im = Image.new('RGB', (total_width, max_height))
                        x_offset = 0
                        for im in imgs:
                            new_im.paste(im, (x_offset, 0))
                            x_offset += im.size[0]
                        new_im.save('%s.png' % img_prefix)
                        for img_path in img_paths:
                            os.remove(img_path)

                    dest_dir = join(output_dir, 'parcellation', parcellation_id)
                    subprocess.call(['cp', '-r', parcellation_dir, dest_dir])

                    breadcrumb_path = os.path.join(dest_dir, 'plots', 'finished.txt')
                    with open(breadcrumb_path, 'w') as f:
                        f.write('Finished')

            # Reset the cache
            shutil.rmtree(tmp_dir_path)
            os.makedirs(tmp_dir_path)


def _get_atlas_paths(
        cfg_path,
        parcellation_ids=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
):
    if isinstance(parcellation_ids, str):
        parcellation_ids = [parcellation_ids]

    if isinstance(reference_atlas_names, str):
        reference_atlas_names = [reference_atlas_names]

    if isinstance(evaluation_atlas_names, str):
        evaluation_atlas_names = [evaluation_atlas_names]

    out = {}
    if not os.path.exists(cfg_path):
        return out

    cfg = get_cfg(cfg_path)
    output_dir = os.path.normpath(cfg['output_dir'])
    compressed = cfg.get('compress_outputs', True)
    suffix = get_suffix(compressed=compressed)

    if parcellation_ids is None:
        parcellation_ids = list(cfg['parcellate'].keys())
    for parcellation_id in parcellation_ids:
        parcellation_dir = get_path(output_dir, 'subdir', 'parcellate', parcellation_id, compressed=compressed)
        if os.path.exists(parcellation_dir):
            out[parcellation_id] = dict(
                reference_atlases={},
                evaluation_atlases={},
                atlases={}
            )
            for filename in [x for x in os.listdir(parcellation_dir) if x.endswith(suffix)]:
                filepath = os.path.realpath(join(parcellation_dir, filename))
                if filename.startswith(REFERENCE_ATLAS_PREFIX):
                    reference_atlas_name = filename[len(REFERENCE_ATLAS_PREFIX):-len(suffix)]
                    if reference_atlas_names is None or reference_atlas_name in reference_atlas_names:
                        out[parcellation_id]['reference_atlases'][reference_atlas_name] = filepath
                elif filename.startswith(EVALUATION_ATLAS_PREFIX):
                    evaluation_atlas_name = filename[len(EVALUATION_ATLAS_PREFIX):-len(suffix)]
                    if evaluation_atlas_names is None or evaluation_atlas_name in evaluation_atlas_names:
                        out[parcellation_id]['evaluation_atlases'][evaluation_atlas_name] = filepath
                elif not filename.startswith('parcellation'):
                    atlas_name = filename[:-len(suffix)]
                    out[parcellation_id]['atlases'][atlas_name] = filepath
            parcellate_kwargs_path = get_path(output_dir, 'kwargs', 'parcellate', parcellation_id, compressed=compressed)
            parcellate_kwargs = get_cfg(parcellate_kwargs_path)
            reference_to_evaluation = {}
            for x in parcellate_kwargs['action_sequence']:
                if x['type'] == 'evaluate':
                    evaluate_kwargs = x['kwargs']
                    evaluation_map = evaluate_kwargs.get(
                        'evaluation_map',
                        None
                    )
                    for reference_atlas in out[parcellation_id]['reference_atlases']:
                        reference_to_evaluation[reference_atlas] = []
                        if evaluation_map is None:
                            reference_to_evaluation[reference_atlas] = out[parcellation_id]['evaluation_atlases'].copy()
                        else:
                            if reference_atlas in evaluation_map:
                                for evaluation_atlas in evaluation_map[reference_atlas]:
                                    reference_to_evaluation[reference_atlas].append(evaluation_atlas)

            out[parcellation_id]['reference_to_evaluation'] = reference_to_evaluation

    return out


def update_atlas_paths(
        atlas_paths,
        dest_dir,
        memo=None
):
    if memo is None:
        memo = set()
    for x in atlas_paths:
        if x == 'reference_to_evaluation':
            continue
        if isinstance(atlas_paths[x], dict):
            update_atlas_paths(atlas_paths[x], dest_dir, memo=memo)
        else:
            if atlas_paths[x] not in memo:
                memo.add(atlas_paths[x])
            new_path = os.path.join(dest_dir, os.path.basename(atlas_paths[x]))
            atlas_paths[x] = new_path

    return memo


def _get_surf_ice_script(
        cfg_paths,
        atlas_paths,
        atlas_names=None,
        atlas_types=None,
        dark_atlas=False,
        subnetwork_id=1,
        min_p=0,
        max_p=1,
        min_act=0,
        max_act=5,
        min_by_path=None,
        max_by_path=None,
        x_res=400,
        y_res=300
):
    if atlas_types is None:
        atlas_types = {'full', 'network', 'subnetworks', 'network_vs_reference', 'network_vs_evaluation'}
    elif isinstance(atlas_types, str):
        atlas_types = {atlas_types}
    else:
        atlas_types = set(atlas_types)
    if min_by_path is None:
        min_by_path = {}
    if max_by_path is None:
        max_by_path = {}

    if not dark_atlas and min_p is None or max_p is None or min_act is None or max_act is None:
        raise ValueError('min_p, max_p, min_act, and max_act must all be specified when using dark_atlas=False')

    script = textwrap.dedent('''\
    import sys
    import os
    import gl

    CWD = os.path.normpath(os.path.join('..', '..', '..', os.getcwd()))

    def get_path(path):
        if not os.path.isabs(path):
            path = os.path.join(CWD, os.path.normpath(path))
        path = os.path.normpath(path)

        return path

    X = %s
    Y = %s

    plot_sets = [
    ''' % (x_res, y_res))

    for cfg_path in cfg_paths:
        if not os.path.exists(cfg_path):
            continue

        cfg = get_cfg(cfg_path)

        for parcellation_id in atlas_paths:
            if subnetwork_id:
                suffix = '_sub%d' % subnetwork_id
            else:
                suffix = ''

            # Full parcellation
            if 'full' in atlas_types:
                plot_set = {}
                output_path = None
                colors = COLORS
                for i, atlas_name in enumerate(atlas_paths[parcellation_id]['atlases']):
                    if not atlas_name.endswith(suffix) or (not suffix and re.match('.*_sub\d+$', atlas_name)):
                        continue
                    reference_atlas_name = atlas_name[:len(atlas_name)-len(suffix)]
                    if atlas_name in atlas_paths[parcellation_id]['atlases']:
                        if output_path is None:
                            output_dir = dirname(
                                atlas_paths[parcellation_id]['reference_atlases'][reference_atlas_name]
                            )
                            output_path = join(
                                output_dir, 'parcellation', parcellation_id,
                                'plots', 'parcellation%s_%%s_%%s.png' % suffix
                            )

                        if i >= len(colors):
                            color = sample_color()
                        else:
                            color = colors[i]

                        color = expand_color(color, base_brightness=BASE_BRIGHTNESS)

                        path = atlas_paths[parcellation_id]['atlases'][atlas_name]
                        plot_set[reference_atlas_name] = dict(
                            name=atlas_name,
                            path=path,
                            output_path=output_path,
                            color=color,
                            min=min_by_path.get(path, min_p),
                            max=max_by_path.get(path, max_p)
                        )
                script += '    %s,\n' % pprint.pformat(plot_set)

            # By network
            if atlas_names is None:
                atlas_names = list(atlas_paths[parcellation_id]['reference_atlases'].keys())
            for atlas_name in atlas_names:
                reference_atlas_name = atlas_name
                atlas_name = atlas_name + suffix
                output_dir = dirname(atlas_paths[parcellation_id]['reference_atlases'][reference_atlas_name])

                # Network
                if 'network' in atlas_types:
                    output_path = join(
                        output_dir, 'parcellation', parcellation_id, 'plots', '%s_%%s_%%s.png' % atlas_name
                    )
                    path = atlas_paths[parcellation_id]['atlases'][atlas_name]
                    plot_set = dict(
                        atlas=dict(
                            name=atlas_name,
                            path=path,
                            output_path=output_path,
                            color=expand_color(BLUE, base_brightness=BASE_BRIGHTNESS),
                            min=min_by_path.get(path, min_p),
                            max=max_by_path.get(path, max_p)
                        ),
                    )
                    script += '    %s,\n' % pprint.pformat(plot_set)

                # Subnetworks
                if 'subnetworks' in atlas_types:
                    output_path = join(
                        output_dir, 'parcellation', parcellation_id, 'plots',
                        '%s_subnetworks_%%s_%%s.png' % reference_atlas_name
                    )
                    subatlases = {}
                    for x in atlas_paths[parcellation_id]['atlases']:
                        if re.sub('_sub\d+$', '_sub', x) == reference_atlas_name + '_sub':
                            ix = int(re.match(reference_atlas_name + '_sub(\d+)', x).group(1))
                            subatlases[ix] = atlas_paths[parcellation_id]['atlases'][x]
                    green = np.linspace(0, 255, len(subatlases)).astype(int)
                    colors = np.stack([[255] * len(green), green, [0] * len(green)], axis=1)
                    colors = colors.tolist()
                    plot_set = {}
                    for i, ix in enumerate(sorted(list(subatlases.keys()))):
                        if i >= len(colors):
                            color = sample_color()
                        else:
                            color = colors[i]
                        path = subatlases[ix]
                        plot_set[ix] = dict(
                            name=reference_atlas_name + '_sub%d' % ix,
                            path=path,
                            output_path=output_path,
                            color=expand_color(color, base_brightness=BASE_BRIGHTNESS),
                            min=min_by_path.get(path, min_p),
                            max=max_by_path.get(path, max_p)
                        )
                    script += '    %s,\n' % pprint.pformat(plot_set)

                # Network vs. reference
                if 'network_vs_reference' in atlas_types:
                    output_path = join(
                        output_dir, 'parcellation', parcellation_id, 'plots',
                        '%s_vs_reference_%%s_%%s.png' % atlas_name
                    )
                    path1 = atlas_paths[parcellation_id]['atlases'][atlas_name]
                    path2 = atlas_paths[parcellation_id]['reference_atlases'][reference_atlas_name]
                    plot_set = dict(
                        atlas=dict(
                            name=atlas_name,
                            path=path1,
                            output_path=output_path,
                            color=expand_color(BLUE, base_brightness=BASE_BRIGHTNESS),
                            min=min_by_path.get(path1, min_p),
                            max=max_by_path.get(path1, max_p)
                        ),
                        reference=dict(
                            name=reference_atlas_name,
                            path=path2,
                            output_path=output_path,
                            color=expand_color(GREEN, base_brightness=BASE_BRIGHTNESS),
                            min=min_by_path.get(path2, min_p),
                            max=max_by_path.get(path2, max_p)
                        ),
                    )
                    script += '    %s,\n' % pprint.pformat(plot_set)

                # Network vs. evaluation
                if 'network_vs_evaluation' in atlas_types:
                    reference_to_evaluation = atlas_paths[parcellation_id]['reference_to_evaluation']
                    for evaluation_atlas_name in reference_to_evaluation.get(reference_atlas_name, []):
                        if evaluation_atlas_name not in atlas_paths[parcellation_id]['evaluation_atlases']:
                            continue
                        output_path = join(
                            output_dir, 'parcellation', parcellation_id,
                            'plots', '%s_vs_%s_%%s_%%s.png' % (atlas_name, evaluation_atlas_name)
                        )
                        path1 = atlas_paths[parcellation_id]['atlases'][atlas_name]
                        path2 = atlas_paths[parcellation_id]['evaluation_atlases'][evaluation_atlas_name]
                        plot_set = dict(
                            atlas=dict(
                                name=atlas_name,
                                path=path1,
                                output_path=output_path,
                                color=expand_color(BLUE, base_brightness=BASE_BRIGHTNESS),
                                min=min_by_path.get(path1, min_p),
                                max=max_by_path.get(path1, max_p)
                            ),
                            evaluation=dict(
                                name=evaluation_atlas_name,
                                path=path2,
                                output_path=output_path,
                                color=expand_color(RED, base_brightness=BASE_BRIGHTNESS),
                                min=min_by_path.get(path2, min_act),
                                max=max_by_path.get(path2, max_act)
                            ),
                        )
                        script += '    %s,\n' % pprint.pformat(plot_set)

    script += ']\n'

    script += 'dark_atlas = %s\n\n' % dark_atlas

    script += textwrap.dedent('''\

    gl.colorbarvisible(0)
    gl.orientcubevisible(0)
    gl.cameradistance(0.55)
    gl.shadername('Default')
    gl.shaderambientocclusion(0.)
    if dark_atlas:
        gl.shaderadjust('Ambient', 0.15)
        gl.shaderadjust('Diffuse', 0.5)
        gl.shaderadjust('Specular', 0.35)
        gl.shaderadjust('SpecularRough', 1.)
        gl.shaderadjust('Edge', 1.)
        gl.shaderlightazimuthelevation(0, 0)
                
    for plot_set in plot_sets:
        for hemi in ('left', 'right'):
            for view in ('lateral', 'medial'):                
                if hemi == 'left':
                    gl.meshload('BrainMesh_ICBM152.lh.mz3')
                    if view == 'lateral':
                        gl.azimuthelevation(-90, 0)
                    else:
                        gl.azimuthelevation(90, 0)
                else:
                    gl.meshload('BrainMesh_ICBM152.rh.mz3')
                    if view == 'lateral':
                        gl.azimuthelevation(90, 0)
                    else:
                        gl.azimuthelevation(-90, 0)
                output_path = None
                colors = None
                
                i = 0
                for atlas_name in plot_set:
                    if output_path is None:
                        output_path = get_path(plot_set[atlas_name]['output_path'])
                    if colors is None:
                        color = plot_set[atlas_name]['color']
                    atlas_path = get_path(plot_set[atlas_name]['path'])
                    min_act = plot_set[atlas_name]['min']
                    max_act = plot_set[atlas_name]['max']

                    if dark_atlas:
                        j_range = range(1, 2)
                    else:
                        j_range = range(1, 5)
                    for j in j_range:
                        overlay = gl.overlayload(atlas_path)
                        gl.overlaycolor(i + 1, *color)
                        if dark_atlas:
                            gl.overlayextreme(i + 1, 3)
                        else:
                            _opacity = int(j / 4 * 100)
                            gl.overlayopacity(i + 1, _opacity)
                        if min_act is not None and max_act is not None:
                            if dark_atlas:
                                _min_act, _max_act = min_act, max_act
                            else:
                                _min_act = min_act + (max_act - min_act) * j / 5
                                _max_act = _min_act
                            gl.overlayminmax(i + 1, _min_act, _max_act)
                        i += 1
                
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                plot_path = output_path % (hemi, view)
                gl.savebmpxy(plot_path, X, Y)
                gl.overlaycloseall()
    exit()
    ''')

    return script


def _is_hemi(path):
    if not path.endswith('.png'):
        return False
    path = path[:-4]
    if not path.endswith('_lateral'):
        if not path.endswith('_medial'):
            return False
        path = path[:-7]
    else:
        path = path[:-8]
    if not path.endswith('_right'):
        if not path.endswith('_left'):
            return False
    return True










######################################
#
#  GROUP ATLAS
#
######################################


def plot_group_atlases(
        cfg_paths,
        parcellation_ids=None,
        atlas_names=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
        plot_dir=join('plots', 'group_atlas')
):
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]

    binary_dir = join(dirname(dirname(__file__)), 'resources', 'surfice', 'Surf_Ice')
    assert os.path.exists(binary_dir), ('Surf Ice directory %s not found. Install using '
        '``python -m parcellate.bin.install_surf_ice``.' % binary_dir)
    binary_path = None
    for path in os.listdir(binary_dir):
        if path in ('surfice', 'surfice.exe'):
            binary_path = join(binary_dir, path)
            break
    assert binary_path, 'No Surf Ice executable found'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    atlases = {}
    atlas_ref = None
    for i, cfg_path in enumerate(cfg_paths):
        stderr('\rRetrieving parcellation %d/%d' % (i + 1, len(cfg_paths)))
        if not os.path.exists(cfg_path):
            continue
        atlas_paths = _get_atlas_paths(
            cfg_path,
            parcellation_ids=parcellation_ids,
            reference_atlas_names=reference_atlas_names,
            evaluation_atlas_names=evaluation_atlas_names
        )

        for parcellation_id in atlas_paths:
            for atlas_name in atlas_paths[parcellation_id]['atlases']:
                if atlas_names is None or atlas_name in atlas_names:
                    atlas_path = atlas_paths[parcellation_id]['atlases'][atlas_name]
                    atlas = image.smooth_img(atlas_path, None)
                    if atlas_ref is None:
                        atlas_ref = atlas
                    atlas = image.get_data(atlas)
                    if parcellation_id not in atlases:
                        atlases[parcellation_id] = {}
                    if atlas_name not in atlases[parcellation_id]:
                        atlases[parcellation_id][atlas_name] = None
                    if atlases[parcellation_id][atlas_name] is None:
                        atlases[parcellation_id][atlas_name] = atlas
                    else:
                        atlases[parcellation_id][atlas_name] += atlas
    stderr('\n')

    data_dir = join(plot_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    atlas_paths = {}
    for parcellation_id in atlases:
        for atlas_name in atlases[parcellation_id]:
            atlas = atlases[parcellation_id][atlas_name]
            atlas /= len(cfg_paths)
            atlas = image.new_img_like(atlas_ref, atlas)
            output_path = join(data_dir, '%s_%s.nii.gz' % (parcellation_id, atlas_name))
            if parcellation_id not in atlas_paths:
                atlas_paths[parcellation_id] = {}
            atlas_paths[parcellation_id][atlas_name] = output_path
            atlas.to_filename(output_path)

    script = _get_surf_ice_script_group(
        atlas_paths,
        plot_dir=plot_dir
    )

    subprocess.call([binary_path, '-S', script])


def _get_surf_ice_script_group(
        atlas_paths,
        plot_dir=join('plots', 'group_atlas')
):
    script = textwrap.dedent('''\
    import sys
    import os
    import gl

    CWD = os.path.normpath(os.path.join('..', '..', '..', os.getcwd()))

    MIN = 0.1

    MAX = 1.

    X = 400
    Y = 300

    ''')

    input_paths = []
    output_paths = []
    colors = []
    for parcellation_id in atlas_paths:
        _plot_dir = join(plot_dir, parcellation_id)
        if not os.path.exists(_plot_dir):
            os.makedirs(_plot_dir)
        atlas_names = sorted(list(atlas_paths[parcellation_id]))
        output_path = join(_plot_dir, 'plots', '%s_group_atlas_%%s_%%s.png' % '_'.join(atlas_names))
        output_paths.append(output_path)
        _input_paths = []
        for atlas_name in atlas_names:
            _input_paths.append(atlas_paths[parcellation_id][atlas_name])
        input_paths.append(_input_paths)
        if len(_input_paths) <= 3:
            _colors = [
                expand_color(RED, base_brightness=BASE_BRIGHTNESS),
                expand_color(BLUE, base_brightness=BASE_BRIGHTNESS),
                expand_color(GREEN, base_brightness=BASE_BRIGHTNESS),
            ]
        else:
            _colors = []
            for _ in range(len(_input_paths)):
                color = np.random.randint(0, 256, size=3)
                color = expand_color(color, base_brightness=BASE_BRIGHTNESS)
                _colors.append(color)
        colors.append(_colors)

    script += 'input_paths = [\n'
    for _input_paths in input_paths:
        script += '    [\n'
        for _input_path in _input_paths:
            script += "        '%s',\n" % _input_path
        script += '    ]\n'
    script += ']\n\n'

    script += 'output_paths = [\n'
    for output_path in output_paths:
        script += "    '%s',\n" % output_path
    script += ']\n\n'

    script += 'colors = [\n'
    for _colors in colors:
        script += '    [\n'
        for _color in _colors:
            script += "        %s,\n" % str(_color)
        script += '    ]\n'
    script += ']\n\n'

    script += textwrap.dedent('''\


    def get_path(path):
        if not os.path.isabs(path):
            path = os.path.join(CWD, os.path.normpath(path))
        path = os.path.normpath(path)

        return path

    for output_path, _input_paths, _colors in zip(output_paths, input_paths, colors):
        output_path = get_path(output_path)

        for hemi in ('left', 'right'):
            for view in ('lateral', 'medial'):
                if hemi == 'left':
                    gl.meshload('BrainMesh_ICBM152.lh.mz3')
                    if view == 'lateral':
                        gl.azimuthelevation(-90, 0)
                    else:
                        gl.azimuthelevation(90, 0)
                else:
                    gl.meshload('BrainMesh_ICBM152.rh.mz3')
                    if view == 'lateral':
                        gl.azimuthelevation(90, 0)
                    else:
                        gl.azimuthelevation(-90, 0)
                        
                for i, (_input_path, _color) in enumerate(zip(_input_paths, _colors)):
                    path = get_path(_input_path)
                    gl.overlayload(_input_path)
                    gl.overlaycolor(i + 1, *_color)
                    gl.overlayminmax(i + 1, MIN, MAX)
                    
                gl.colorbarvisible(0)
                gl.orientcubevisible(0)
                gl.cameradistance(0.55)

                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                plot_path = output_path % (hemi, view)
                gl.savebmpxy(plot_path, X, Y)
    ''')

    return script





######################################
#
#  PERFORMANCE
#
######################################


def plot_performance(
        cfg_paths,
        parcellation_ids=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
        baseline_atlas_names=None,
        include_thresholds=False,
        plot_dir=join('plots', 'performance'),
        reference_atlas_name_to_label=REFERENCE_ATLAS_NAME_TO_LABEL,
        dump_data=False
):
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]
    if isinstance(parcellation_ids, str):
        parcellation_ids = [parcellation_ids]
    if isinstance(baseline_atlas_names, str):
        baseline_atlas_names = [baseline_atlas_names]

    dfs = {}
    for cfg_path in cfg_paths:
        if not os.path.exists(cfg_path):
            continue
        cfg = get_cfg(cfg_path)
        if parcellation_ids is None:
            _parcellation_ids = list(cfg['parcellate'].keys())
        else:
            _parcellation_ids = parcellation_ids
        output_dir = cfg['output_dir']
        for parcellation_id in _parcellation_ids:
            df_path = get_path(output_dir, 'evaluation', 'parcellate', parcellation_id)
            if os.path.exists(df_path):
                if parcellation_id not in dfs:
                    dfs[parcellation_id] = []
                df = pd.read_csv(df_path)
                df['cfg_path'] = cfg_path
                dfs[parcellation_id].append(df)

    for parcellation_id in dfs:
        if not (parcellation_id in dfs and len(dfs[parcellation_id])):
            continue
        df = pd.concat(dfs[parcellation_id], axis=0)
        atlas_names = df[df.parcel_type != 'baseline'].parcel.unique().tolist()
        _reference_atlas_names = df['%sname' % REFERENCE_ATLAS_PREFIX].unique().tolist()
        if reference_atlas_names is None:
            _reference_atlas_names = df['%sname' % REFERENCE_ATLAS_PREFIX].unique().tolist()
        else:
            _reference_atlas_names = [x for x in reference_atlas_names if x in _reference_atlas_names]

        for atlas_name in atlas_names:
            # Similarity to reference
            reference_atlas_name = re.sub('_sub\d+$', '', atlas_name)
            _df = df[(df.parcel == atlas_name)]
            cols = ['%s%s_score' % (REFERENCE_ATLAS_PREFIX, x) for x in _reference_atlas_names]
            __df = _df[_df['%sname' % REFERENCE_ATLAS_PREFIX] == \
                       reference_atlas_name][cols].rename(_rename_performance, axis=1)
            colors = ['m']
            xlabel = None
            ylabel = 'Similarity to Reference'
            fig = _plot_performance(
                __df,
                colors=colors,
                xlabel=xlabel,
                ylabel=ylabel,
                divider=False
            )
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            fig.savefig(join(plot_dir, '%s_%ssim.png' % (atlas_name, REFERENCE_ATLAS_PREFIX)), dpi=300)
            if dump_data:
                __df.to_csv(
                    join(plot_dir, '%s_%ssim.csv' % (atlas_name, REFERENCE_ATLAS_PREFIX)),
                    index=False
                )

            suffixes = ['']
            if include_thresholds:
                suffixes += list(SUFFIX2NAME.keys())

            _baseline_atlas_names = baseline_atlas_names
            if _baseline_atlas_names is None:
                _baseline_atlas_names = ['%s%s' % (REFERENCE_ATLAS_PREFIX, reference_atlas_name)]

            labels = [
                reference_atlas_name_to_label.get(baseline_atlas_name, baseline_atlas_name) for
                         baseline_atlas_name in _baseline_atlas_names
            ] + ['FC']

            # Similarity to evaluation
            __df = _df[_df['%sname' % REFERENCE_ATLAS_PREFIX] == \
                       reference_atlas_name]
            _evaluation_atlas_names = [x[:-6] for x in __df if x.endswith('_score') and \
                                       x.startswith(EVALUATION_ATLAS_PREFIX) and \
                                       (evaluation_atlas_names is None or x[:-6] in evaluation_atlas_names) and \
                                       np.isfinite(__df[x].values).sum() > 0]
            cols = ['%s_score' % x for x in _evaluation_atlas_names]
            cols = [x for x in cols if x in _df]
            __df = __df[cols].rename(_rename_performance, axis=1)
            has_finite = np.isfinite(__df.values).sum() > 0
            if has_finite:
                dfb = []
                for baseline_atlas_name in _baseline_atlas_names:
                    _dfb = df[df.parcel == baseline_atlas_name]
                    _dfb = _dfb[_dfb['%sname' % REFERENCE_ATLAS_PREFIX] == \
                                reference_atlas_name][cols].rename(_rename_performance, axis=1)
                    dfb.append(_dfb)
                ylabel = 'Similarity to evaluation atlas'
                xlabel = None
                colors = ['c', 'm']
                fig = _plot_performance(
                    *dfb, __df,
                    colors=colors,
                    labels=labels,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    divider=False
                )
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig.savefig(join(plot_dir, '%s_%ssim.png' % (
                    atlas_name, EVALUATION_ATLAS_PREFIX)), dpi=300)
                if dump_data:
                    csv = dfb + [__df]
                    for i, _csv in enumerate(csv):
                        _csv['label'] = labels[i]
                    csv = pd.concat(csv, axis=0)
                    csv.to_csv(
                        join(plot_dir, '%s_%ssim.csv' % (atlas_name, EVALUATION_ATLAS_PREFIX)),
                        index=False
                    )

            # Evaluation contrast size
            cols = ['%s_contrast' % x for x in _evaluation_atlas_names]
            cols = [x for x in cols if x in _df]
            __df = _df[_df['%sname' % REFERENCE_ATLAS_PREFIX] == \
                       reference_atlas_name][cols].rename(_rename_performance, axis=1)
            has_finite = np.isfinite(__df.values).sum() > 0
            if has_finite:
                dfb = []
                for baseline_atlas_name in _baseline_atlas_names:
                    _dfb = df[df.parcel == baseline_atlas_name]
                    _dfb = _dfb[_dfb['%sname' % REFERENCE_ATLAS_PREFIX] == \
                                reference_atlas_name][cols].rename(_rename_performance, axis=1)
                    dfb.append(_dfb)
                ylabel = 'Contrast'
                xlabel = None
                colors = ['c', 'm']
                fig = _plot_performance(
                    *dfb, __df,
                    colors=colors,
                    labels=labels,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    divider=False
                )
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig.savefig(join(plot_dir, '%s_%scontrast.png' % (
                    atlas_name, EVALUATION_ATLAS_PREFIX)), dpi=300)
                if dump_data:
                    csv = dfb + [__df]
                    for i, _csv in enumerate(csv):
                        _csv['label'] = labels[i]
                    csv = pd.concat(csv, axis=0)
                    csv.to_csv(
                        join(plot_dir, '%s_%scontrast.csv' % (atlas_name, EVALUATION_ATLAS_PREFIX)),
                        index=False
                    )


def _plot_performance(
        *dfs,
        colors=None,
        labels=None,
        xlabel=None,
        tick_labels=None,
        ylabel=None,
        divider=False,
        width=None,
        height=3
):
    plt.close('all')
    n_colors = len(dfs)
    bar_width = 0.8 / n_colors
    if tick_labels is None:
        n_ticks = None
    else:
        n_ticks = len(tick_labels)
    x = None
    xlim = None
    spacer = 1
    while len(colors) < n_colors:
        colors = [None] + colors
    for i, df in enumerate(dfs):
        if n_ticks is None:
            n_ticks = len(df.columns)
        if tick_labels is None:
            tick_labels = df.columns.tolist()
        if n_ticks == 1:
            divider = False
        y = df.mean(axis=0)
        yerr = df.sem(axis=0)
        if x is None:
            if divider:
                xpad = spacer
                x = np.concatenate([np.zeros(1), np.arange(1, n_ticks) + spacer])
            else:
                xpad = 1
                x = np.arange(n_ticks)
        if xlim is None:
            xlim = (x.min() - xpad, x.max() + xpad)
        _x = x + (i - (n_colors - 1) / 2) * bar_width
        if i >= len(colors):
            color = sample_color()
        else:
            color = colors[i]
        if labels is not None and i < len(labels):
            label = labels[i]
        else:
            label = None

        plt.bar(_x, y, width=bar_width, color=color, label=label)
        if len(df) > 1:
            plt.errorbar(_x, y, yerr=yerr, fmt='none', color=color)

        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
    plt.xticks(
        x,
        tick_labels,
        rotation=45,
        ha='right',
        rotation_mode='anchor'
    )
    plt.xlim(xlim)
    if labels is not None:
        legend_kwargs = dict(
            loc='lower center',
            bbox_to_anchor=(0.5, 1.1),
            ncols=n_colors,
            frameon=False,
            fancybox=False,
        )
        plt.legend(**legend_kwargs)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().axhline(y=0, lw=1, c='k', alpha=1)
    if width is None:
        width = n_ticks * 0.05 * height + 0.7 * height
    plt.gcf().set_size_inches(width, height)
    if divider:
        loc = (1 + spacer) / 2
        plt.gca().axvline(loc, color='k', lw=1)
    plt.tight_layout()

    return plt.gcf()


def _rename_performance(x):
    for suffix in SUFFIX2NAME:
        if x.endswith(suffix):
            return SUFFIX2NAME[suffix]
    return x.replace(REFERENCE_ATLAS_PREFIX, '')\
            .replace(EVALUATION_ATLAS_PREFIX, '')\
            .replace('_score', '')\
            .replace('_contrast', '')


def plot_performance_by_data_size(
        cfg_paths,
        parcellation_ids=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
        plot_dir=join('plots', 'performance_by_data_size'),
        dump_data=False
):
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]
    if isinstance(parcellation_ids, str):
        parcellation_ids = [parcellation_ids]

    dfs = {}
    for cfg_path in cfg_paths:
        if not os.path.exists(cfg_path):
            continue
        cfg = get_cfg(cfg_path)
        if parcellation_ids is None:
            _parcellation_ids = list(cfg['parcellate'].keys())
        else:
            _parcellation_ids = parcellation_ids
        output_dir = cfg['output_dir']
        for parcellation_id in _parcellation_ids:
            df_path = get_path(output_dir, 'evaluation', 'parcellate', parcellation_id)
            if os.path.exists(df_path):
                if parcellation_id not in dfs:
                    dfs[parcellation_id] = []

                parcellate_cfg_path = get_path(output_dir, 'kwargs', 'parcellate', parcellation_id)
                parcellate_cfg = get_cfg(parcellate_cfg_path)
                sample_id = get_action_attr('sample', parcellate_cfg['action_sequence'], 'id')
                tr = get_action_attr('sample', parcellate_cfg['action_sequence'], 'kwargs').get('tr', 2)
                n_trs = pd.read_csv(get_path(output_dir, 'metadata', 'sample', sample_id)).n_trs.unique().tolist()
                assert len(n_trs) == 1, 'Should have only one value for n_trs, got %d.' % len(n_trs)
                n_trs = n_trs[0]
                minutes = (n_trs * tr) / 60

                df = pd.read_csv(df_path)
                df['cfg_path'] = cfg_path
                df['minutes'] = minutes
                dfs[parcellation_id].append(df)

    for parcellation_id in dfs:
        if not (parcellation_id in dfs and len(dfs[parcellation_id])):
            continue
        df = pd.concat(dfs[parcellation_id], axis=0)
        atlas_names = df[df.parcel_type != 'baseline'].parcel.unique().tolist()
        _reference_atlas_names = df['%sname' % REFERENCE_ATLAS_PREFIX].unique().tolist()
        if reference_atlas_names is None:
            _reference_atlas_names = df['%sname' % REFERENCE_ATLAS_PREFIX].unique().tolist()
        else:
            _reference_atlas_names = [x for x in reference_atlas_names if x in _reference_atlas_names]

        for atlas_name in atlas_names:
            reference_atlas_name = re.sub('_sub\d+$', '', atlas_name)
            # Similarity to reference
            _df = df[(df.parcel == atlas_name)]
            cols = ['%sscore' % REFERENCE_ATLAS_PREFIX, 'minutes']
            __df = _df[_df['%sname' % REFERENCE_ATLAS_PREFIX] == \
                       reference_atlas_name][cols].rename(_rename_performance, axis=1)
            xlabel = 'Minutes of fMRI data'
            ylabel = 'Similarity to Reference'
            fig = _plot_performance_by_data_size(
                __df,
                xlabel=xlabel,
                ylabel=ylabel
            )
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            fig.savefig(
                join(plot_dir, '%s_%ssim_by_data_size.png' % (atlas_name, REFERENCE_ATLAS_PREFIX)),
                dpi=300
            )
            if dump_data:
                __df.to_csv(
                    join(plot_dir, '%s_%s_sim_by_data_size.csv' % (atlas_name, REFERENCE_ATLAS_PREFIX)),
                    index=False
                )

            _evaluation_atlas_names = [x[:-6] for x in df if x.endswith('_score') and \
                                       x.startswith(EVALUATION_ATLAS_PREFIX)]
            if evaluation_atlas_names is not None:
                _evaluation_atlas_names = [x for x in evaluation_atlas_names if x in _evaluation_atlas_names]

            for evaluation_atlas_name in _evaluation_atlas_names:
                # Similarity to evaluation
                cols = ['%s_score' % evaluation_atlas_name, 'minutes']
                cols = [x for x in cols if x in _df]
                __df = _df[_df['%sname' % REFERENCE_ATLAS_PREFIX] == \
                           reference_atlas_name][cols].rename(_rename_performance, axis=1)
                if len(__df) == 0 or np.any(np.isfinite(__df.values).sum(axis=0) == 0):
                    continue
                xlabel = 'Minutes of fMRI data'
                ylabel = 'Similarity to %s' % evaluation_atlas_name
                fig = _plot_performance_by_data_size(
                    __df,
                    xlabel=xlabel,
                    ylabel=ylabel
                )
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig.savefig(join(plot_dir, '%s_%s_sim_by_data_size.png' % (
                    atlas_name, evaluation_atlas_name)), dpi=300)
                if dump_data:
                    csv = __df
                    csv.to_csv(
                        join(plot_dir, '%s_%s_sim_by_data_size.csv' % (atlas_name, evaluation_atlas_name)),
                        index=False
                    )

                # Evaluation contrast size
                cols = ['%s_contrast' % evaluation_atlas_name, 'minutes']
                cols = [x for x in cols if x in _df]
                __df = _df[_df['%sname' % REFERENCE_ATLAS_PREFIX] == \
                           reference_atlas_name][cols].rename(_rename_performance, axis=1)
                xlabel = 'Minutes of fMRI data'
                ylabel = '%s Contrast' % evaluation_atlas_name
                fig = _plot_performance_by_data_size(
                    __df,
                    xlabel=xlabel,
                    ylabel=ylabel
                )
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig.savefig(join(plot_dir, '%s_%s_contrast_by_data_size.png' % (
                    atlas_name, evaluation_atlas_name)), dpi=300)
                if dump_data:
                    csv = __df
                    csv = pd.concat(csv, axis=0)
                    csv.to_csv(
                        join(plot_dir, '%s_%s_contrast_by_data_size.csv' % (atlas_name, evaluation_atlas_name)),
                        index=False
                    )


def _plot_performance_by_data_size(
        df,
        color='m',
        xlabel=None,
        ylabel=None,
        width=4,
        height=3,
):
    x = df.minutes.values
    df = df[[col for col in df if col != 'minutes']]
    y = df.values[..., 0]

    sel = np.isfinite(y)
    x = x[sel]
    y = y[sel]

    m, b = np.polyfit(x, y, 1)
    reg_x = np.linspace(x.min(), x.max(), 500)
    reg_y = reg_x * m + b
    r = np.corrcoef(x, y)[0, 1]
    n = len(y)
    t = r * np.sqrt((n - 2) / (1 - r**2))
    p = 2 * (1 - stats.t(n - 2).cdf(np.abs(t)))
    n_stars = 0
    if p < 0.001:
        n_stars = 3
    elif p < 0.01:
        n_stars = 2
    elif p < 0.05:
        n_stars = 1

    plt.close('all')
    plt.scatter(x, y, color=color, marker='.', s=10, linewidth=0, zorder=1, alpha=0.2)
    plt.plot(reg_x, reg_y, color=color, linestyle='solid', zorder=2)
    plt.text(
        1, 1, 'r = %0.2f%s' % (r, '*' * n_stars), horizontalalignment='right', verticalalignment='top',
        transform=plt.gca().transAxes, fontsize=12
    )

    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.ylim(tuple(np.quantile(y, [0.05, 0.95])))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().axhline(y=0, lw=1, c='k', alpha=1)
    plt.gcf().set_size_inches(width, height)
    plt.tight_layout()

    return plt.gcf()






######################################
#
#  GRID
#
######################################


def plot_grid(
        cfg_paths,
        aggregation_ids=None,
        dimensions=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
        baseline_atlas_names=None,
        plot_selected=False,
        plot_dir=join('plots', 'grid'),
        reference_atlas_name_to_label=REFERENCE_ATLAS_NAME_TO_LABEL,
        dump_data=False
):
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]
    if isinstance(aggregation_ids, str):
        aggregation_ids = [aggregation_ids]
    if isinstance(dimensions, str):
        dimensions = [dimensions]
    if isinstance(reference_atlas_names, str):
        reference_atlas_names = [reference_atlas_names]
    if isinstance(evaluation_atlas_names, str):
        evaluation_atlas_names = [evaluation_atlas_names]
    if isinstance(baseline_atlas_names, str):
        baseline_atlas_names = [baseline_atlas_names]

    dfs = {}
    grid_params = None
    for cfg_path in cfg_paths:
        if not os.path.exists(cfg_path):
            continue
        cfg = get_cfg(cfg_path)
        if grid_params is None:
            grid_params = cfg['grid']
        if aggregation_ids is None:
            _aggregation_ids = list(cfg['parcellate'].keys())
        else:
            _aggregation_ids = aggregation_ids
        output_dir = cfg['output_dir']
        for aggregation_id in _aggregation_ids:
            df_path = get_path(output_dir, 'evaluation', 'aggregate', aggregation_id)
            evaluation_id = None
            action_sequence = get_action_sequence(cfg, 'aggregate', aggregation_id)
            for action in action_sequence:
                if action['type'] == 'evaluate':
                    evaluation_id = action['id']
                    break
            if os.path.exists(df_path):
                print(df_path)
                if aggregation_id not in dfs:
                    dfs[aggregation_id] = []
                df = pd.read_csv(df_path)
                grid_ids = df.grid_id.unique()
                evaluation_df = []
                for grid_id in grid_ids:
                    if evaluation_id:
                        output_dir_ = join(output_dir, 'grid', grid_id)
                        evaluation_path = get_path(output_dir_, 'output', 'evaluate', evaluation_id)
                        if os.path.exists(evaluation_path):
                            df_ = pd.read_csv(evaluation_path)
                            evaluation_cols = ['parcel', 'ref_name']
                            for col in df_:
                                if col not in df:
                                    is_score = col.startswith(EVALUATION_ATLAS_PREFIX) and col.endswith('_score')
                                    is_contrast = col.startswith(EVALUATION_ATLAS_PREFIX) and col.endswith('_contrast')
                                    if is_score or is_contrast:
                                        if is_score:
                                            postfix_len = 6
                                        else:
                                            postfix_len = 9
                                        atlas_name = col[:-postfix_len]
                                        if evaluation_atlas_names is None or atlas_name in evaluation_atlas_names:
                                            evaluation_cols.append(col)
                            df_ = df_[evaluation_cols]
                            df_['grid_id'] = grid_id
                            evaluation_df.append(df_)
                evaluation_df = pd.concat(evaluation_df, axis=0)
                df = pd.merge(df, evaluation_df, how='outer', on=['parcel', 'ref_name', 'grid_id'])
                df['cfg_path'] = cfg_path
                dfs[aggregation_id].append(df)

    _, grid_dict = get_grid_array_from_grid_params(grid_params)
    _dimensions = list(grid_dict.keys())
    _dimensions_all = _dimensions
    if dimensions:
        _dimensions = [x for x in dimensions if x in _dimensions]

    for aggregation_id in dfs:
        if not (aggregation_id in dfs and len(dfs[aggregation_id])):
            continue
        df = pd.concat(dfs[aggregation_id], axis=0)
        atlas_names = df[df.parcel_type != 'baseline'].parcel.unique().tolist()
        _reference_atlas_names = df['%sname' % REFERENCE_ATLAS_PREFIX].unique().tolist()
        if reference_atlas_names is None:
            _reference_atlas_names = df['%sname' % REFERENCE_ATLAS_PREFIX].unique().tolist()
        else:
            _reference_atlas_names = [x for x in reference_atlas_names if x in _reference_atlas_names]

        for atlas_name in atlas_names:
            reference_atlas_name = re.sub('_sub\d+$', '', atlas_name)
            labels = ['FC', reference_atlas_name_to_label.get(reference_atlas_name, reference_atlas_name)]

            for dimension in _dimensions:
                _dimensions_other = [x for x in _dimensions if x != dimension]
                for _dimension in [dimension] + _dimensions_other:
                    if _dimension not in df:
                        df[_dimension] = df.grid_id.apply(_get_param_value, args=(_dimension,))
                # Similarity to reference
                perf_col = '%sscore' % REFERENCE_ATLAS_PREFIX
                _df = df[(df.parcel == atlas_name)]
                _df = _df[_df['%sname' % REFERENCE_ATLAS_PREFIX] == \
                          reference_atlas_name]
                if plot_selected:
                    selected = _df[_df.selected]
                    missing = False
                    for x in (dimension, perf_col):
                        if not x in list(selected.columns):
                            missing = True
                            break
                    if missing:
                        continue
                    selected = selected[[dimension, perf_col]]
                    selected = selected.set_index(dimension)[perf_col]
                else:
                    selected = None
                __df = _df.pivot(
                    columns=[dimension] + _dimensions_other,
                    index='cfg_path',
                    values=perf_col
                )
                idx_names = [_rename_grid(x) for x in __df.columns.names]
                __df.columns = __df.columns.set_names(idx_names)

                fig = _plot_grid(
                    __df,
                    _rename_grid(dimension),
                    labels=labels,
                    selected=selected,
                    colors=['m'],
                    ylabel=_rename_grid(perf_col)
                )

                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig.savefig(
                    join(plot_dir, '%s_%ssim_by_%s_grid.png' % (atlas_name, REFERENCE_ATLAS_PREFIX, dimension)),
                    dpi=300
                )
                if dump_data:
                    __df.to_csv(
                        join(plot_dir, '%s_%ssim_by_%s_grid.csv' % (atlas_name, REFERENCE_ATLAS_PREFIX, dimension)),
                        index=False
                    )

                # Similarity to evaluation
                _dfr = df[df.parcel == '%s%s' % (REFERENCE_ATLAS_PREFIX, reference_atlas_name)]
                _dfr = _dfr[_dfr['%sname' % REFERENCE_ATLAS_PREFIX] == \
                            reference_atlas_name]
                _evaluation_atlas_names = [x[:-6] for x in df if x.endswith('_score') and
                                           x.startswith(EVALUATION_ATLAS_PREFIX) and
                                           (evaluation_atlas_names is None or x[:-6] in evaluation_atlas_names)]
                for evaluation_atlas_name in _evaluation_atlas_names:
                    perf_col = '%s_score' % evaluation_atlas_name
                    if plot_selected:
                        selected = _df[_df.selected][[dimension, perf_col]]
                        selected = selected.set_index(dimension)[perf_col]
                    else:
                        selected = None
                    __df = _df.pivot(
                        columns=[dimension] + _dimensions_other,
                        index='cfg_path',
                        values=perf_col
                    )
                    idx_names = [_rename_grid(x) for x in __df.columns.names]
                    __df.columns = __df.columns.set_names(idx_names)
                    if len(__df) == 0 or np.any(np.isfinite(__df.values).sum(axis=0) == 0):
                        continue

                    _baseline_atlas_names = baseline_atlas_names
                    if _baseline_atlas_names is None:
                        _baseline_atlas_names = ['%s%s' % (REFERENCE_ATLAS_PREFIX, reference_atlas_name)]

                    labels = [
                                 reference_atlas_name_to_label.get(baseline_atlas_name, baseline_atlas_name) for
                                 baseline_atlas_name in _baseline_atlas_names
                             ] + ['FC']

                    dfb = []
                    for baseline_atlas_name in _baseline_atlas_names:
                        _dfb = df[df.parcel == baseline_atlas_name]
                        _dfb = _dfb[
                            (_dfb['%sname' % REFERENCE_ATLAS_PREFIX] == reference_atlas_name) &
                            (_dfb['ref_name'] == reference_atlas_name)
                        ]
                        _dfb = _dfb.pivot(
                            columns=[dimension] + _dimensions_other,
                            index='cfg_path',
                            values=perf_col
                        )
                        idx_names = [_rename_grid(x) for x in _dfb.columns.names]
                        _dfb.columns = _dfb.columns.set_names(idx_names)
                        dfb.append(_dfb)

                    fig = _plot_grid(
                        __df,
                        _rename_grid(dimension),
                        dfb=dfb,
                        labels=labels,
                        selected=selected,
                        colors=['gray', 'm'],
                        ylabel=_rename_grid(perf_col)
                    )

                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    fig.savefig(
                        join(plot_dir, '%s_%s_by_%s_sim_grid.png' % (atlas_name, evaluation_atlas_name, dimension)),
                        dpi=300
                    )
                    if dump_data:
                        csv = dfb + [__df]
                        for i, _csv in enumerate(csv):
                            _csv['label'] = labels[i]
                        csv = pd.concat(csv, axis=0)
                        csv.to_csv(
                            join(plot_dir, '%s_%s_by_%s_sim_grid.csv' % (atlas_name, evaluation_atlas_name, dimension)),
                            index=False
                        )

                    perf_col = '%s_contrast' % evaluation_atlas_name
                    if plot_selected:
                        selected = _df[_df.selected][[dimension, perf_col]]
                        selected = selected.set_index(dimension)[perf_col]
                    else:
                        selected = None
                    __df = _df.pivot(
                        columns=[dimension] + _dimensions_other,
                        index='cfg_path',
                        values=perf_col
                    )
                    idx_names = [_rename_grid(x) for x in __df.columns.names]
                    __df.columns = __df.columns.set_names(idx_names)

                    dfb = []
                    for baseline_atlas_name in _baseline_atlas_names:
                        _dfb = df[df.parcel == baseline_atlas_name]
                        _dfb = _dfb[_dfb['%sname' % REFERENCE_ATLAS_PREFIX] == \
                                    reference_atlas_name]
                        _dfb = _dfb.pivot(
                            columns=[dimension] + _dimensions_other,
                            index='cfg_path',
                            values=perf_col
                        )
                        idx_names = [_rename_grid(x) for x in _dfb.columns.names]
                        _dfb.columns = _dfb.columns.set_names(idx_names)
                        dfb.append(_dfb)

                    fig = _plot_grid(
                        __df,
                        _rename_grid(dimension),
                        dfb=dfb,
                        labels=labels,
                        selected=selected,
                        colors=['gray', 'm'],
                        ylabel=_rename_grid(perf_col)
                    )

                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    fig.savefig(
                        join(plot_dir, '%s_%s_by_%s_contrast_grid.png' %
                             (atlas_name, evaluation_atlas_name, dimension)),
                        dpi=300
                    )
                    if dump_data:
                        csv = dfb + [__df]
                        for i, _csv in enumerate(csv):
                            _csv['label'] = labels[i]
                        csv = pd.concat(csv, axis=0)
                        csv.to_csv(
                            join(plot_dir, '%s_%s_by_%s_contrast_grid.csv' %
                                 (atlas_name, evaluation_atlas_name, dimension)),
                            index=False
                        )


def _plot_grid(
        df,
        dimension,
        dfb=None,
        labels=None,
        selected=None,
        colors=None,
        ylabel=None,
        width=4,
        height=3,
):
    plt.close('all')
    xlabel = dimension
    ticks = None
    tick_labels = None
    tick_map = {}

    dfs = []
    if dfb is not None:
        if not isinstance(dfb, list):
            dfs.append(dfb)
        else:
            dfs += dfb
    dfs.append(df)
    i = 0
    while len(colors) < len(dfs):
        colors = [None] + colors
    for i, _df in enumerate(dfs):
        label = labels[i]
        x = _df.columns
        if len(x.names) > 1:
            stack_levels = [z for z in _df.columns.names if z != dimension]
            _df = _df.stack(level=stack_levels, future_stack=True)
            x = _df.columns.get_level_values(level=dimension)
        else:
            assert x.name == dimension, 'Mismatch between dimension (%s) and columns index name (%s)' % \
                                        (dimension, x.name)
        if not pd.api.types.is_numeric_dtype(x.dtype):
            ticks = np.arange(len(x))
            tick_labels = x.values
            tick_map = {x: y for x, y in zip(tick_labels, ticks)}
            x = ticks
        y = _df.mean(axis=0)
        yerr = _df.sem(axis=0)
        if i >= len(colors):
            color = sample_color()
        else:
            color = colors[i]

        if len(_df) > 1:
            plt.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.2, linewidth=0, zorder=i)
        if i == (len(dfs) - 1):
            linestyle = 'solid'
        else:
            linestyle = 'dotted'
        plt.plot(x, y, color=color, linestyle=linestyle, zorder=i, label=label)

    if selected is not None and len(selected):
        index = pd.Series(selected.index)
        if ticks is not None:
            index = index.map(tick_map)
        x = index.mean()
        xerr = index.sem()
        if not np.isfinite(xerr):
            xerr = None
        y = selected.mean()
        yerr = selected.sem()
        if not np.isfinite(yerr):
            yerr = None
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='c.', linewidth=1, markersize=3, zorder=i+1)

    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if ticks is not None:
        plt.xticks(
            ticks,
            tick_labels,
            rotation=45,
            ha='right',
            rotation_mode='anchor'
        )

    if labels is not None and len(labels) > 1:
        legend_kwargs = dict(
            loc='lower center',
            bbox_to_anchor=(0.5, 1.1),
            ncols=len(labels),
            frameon=False,
            fancybox=False,
        )
        plt.legend(**legend_kwargs)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().axhline(y=0, lw=1, c='k', alpha=1)
    plt.gcf().set_size_inches(width, height)
    plt.tight_layout()

    return plt.gcf()


def _get_param_value(s, grid_id):
    val = re.search('%s([^_]+)' % grid_id, s).group(1)
    try:
        val_i = int(val)
        val_f = float(val)
        if val_i == val_f:
            return val_i
        return val_f
    except ValueError:
        pass

    return val


def _rename_grid(x):
    for suffix in SUFFIX2NAME:
        if x.endswith(suffix):
            return SUFFIX2NAME[suffix]
    if x == 'n_networks':
        return 'N Networks'
    if x == '%sscore' % REFERENCE_ATLAS_PREFIX:
        return 'Similarity to Reference'
    x = x.replace(REFERENCE_ATLAS_PREFIX, '')\
         .replace(EVALUATION_ATLAS_PREFIX, '')
    if x.endswith('_score'):
        return 'Similarity to %s' % x[:-6]
    if x.endswith('_contrast'):
        return '%s Contrast' % x[:-9]
    return x






######################################
#
#  EXECUTABLE
#
######################################

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot parcellation performance')
    argparser.add_argument('cfg_paths', nargs='+', help=textwrap.dedent('''\
        Path(s) to parcellate config files (config.yml) to plot.'''
    ))
    argparser.add_argument('-t', '--plot_type', nargs='+', default=['performance'], help=textwrap.dedent('''\
        Type of plot to generate. One of ``atlas``, ``group_atlas``, ``performance``, ``performance_by_data_size``, 
        ``grid``, or ``all``. Defaults to ["performance"].
    '''))
    argparser.add_argument('-p', '--parcellation_ids', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of parcellation(s) to use for plotting. If None, use all available parcellations.
    '''))
    argparser.add_argument('-a', '--aggregation_ids', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of aggregation(s) to use for plotting. If None, use all available parcellations.
    '''))
    argparser.add_argument('-A', '--atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of parcellation atlas(es) to use for plotting. If None, use all available parcellation atlases.'''
    ))
    argparser.add_argument('-r', '--reference_atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of reference atlas(es) to use for plotting. If None, use all available reference atlases.'''
    ))
    argparser.add_argument('-e', '--evaluation_atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of evaluation atlas(es) to use for plotting. If None, use all available evaluation atlases.'''
    ))
    argparser.add_argument('-b', '--baseline_atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of atlas(es) to use as baseline for plotting. If None, use all reference atlas.'''
    ))
    argparser.add_argument('-d', '--dimensions', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of grid-searched dimension(s) to plot. If None, use all available dimensions.'''
    ))
    argparser.add_argument('-i', '--subnetwork_id', type=int, default=1, help=textwrap.dedent('''\
        Index of subnetwork_id to use where relevant for atlas construction (e.g., comparison to reference).
        Value 0 plots the aggregate across subnetworks.
    '''))
    argparser.add_argument('-T', '--include_thresholds', action='store_true', help=textwrap.dedent('''\
        Include performance when binarizing the atlas at different probability thresholds.'''
    ))
    argparser.add_argument('--atlas_types', nargs='+', default=None, help=textwrap.dedent('''\
        Types of atlases to plot if ``atlas`` is in ``plot_type``. Any combination of ``full``, ``network``,
         ``subnetworks``, ``network_vs_reference``, and ``network_vs_evaluation``. If None, use all available
         atlas types.'''
    ))
    argparser.add_argument('--dark_atlas', action='store_true', help=textwrap.dedent('''\
        Plot atlases against a dark background brain. Happens to run a lot faster this way.'''
    ))
    argparser.add_argument('-D', '--dump_data', action='store_true', help=textwrap.dedent('''\
        Save plot data to CSV.'''
    ))
    argparser.add_argument('-O', '--overwrite_atlases', action='store_true', help=textwrap.dedent('''\
        Overwrite existing atlas plots. If not used, previously completed plot sets will be skipped.'''
    ))
    argparser.add_argument('-o', '--output_dir', default='plots', help=textwrap.dedent('''\
        Output directory for performance and grid plots (atlases are saved in each model directory).
    '''))
    args = argparser.parse_args()

    cfg_paths = args.cfg_paths
    plot_type = set(args.plot_type)
    parcellation_ids = args.parcellation_ids
    aggregation_ids = args.aggregation_ids
    atlas_names = args.atlas_names
    reference_atlas_names = args.reference_atlas_names
    evaluation_atlas_names = args.evaluation_atlas_names
    baseline_atlas_names = args.baseline_atlas_names
    dimensions = args.dimensions
    subnetwork_id = args.subnetwork_id
    if not subnetwork_id:
        subnetwork_id = None
    include_thresholds = args.include_thresholds
    atlas_types = args.atlas_types
    dark_atlas = args.dark_atlas
    dump_data = args.dump_data
    overwrite_atlases = args.overwrite_atlases
    output_dir = args.output_dir

    if plot_type & {'atlas', 'all'}:
        plot_atlases(
            cfg_paths,
            parcellation_ids=parcellation_ids,
            subnetwork_id=subnetwork_id,
            atlas_names=atlas_names,
            reference_atlas_names=reference_atlas_names,
            evaluation_atlas_names=evaluation_atlas_names,
            atlas_types=atlas_types,
            dark_atlas=dark_atlas,
            overwrite_atlases=overwrite_atlases
        )
    if plot_type & {'group_atlas', 'all'}:
        plot_group_atlases(
            cfg_paths,
            parcellation_ids=parcellation_ids,
            atlas_names=atlas_names,
            reference_atlas_names=reference_atlas_names,
            evaluation_atlas_names=evaluation_atlas_names
        )
    if plot_type & {'performance', 'all'}:
        plot_performance(
            cfg_paths,
            parcellation_ids=parcellation_ids,
            reference_atlas_names=reference_atlas_names,
            evaluation_atlas_names=evaluation_atlas_names,
            baseline_atlas_names=baseline_atlas_names,
            include_thresholds=include_thresholds,
            plot_dir=join(output_dir, 'performance'),
            dump_data=dump_data
        )
    if plot_type & {'performance_by_data_size', 'all'}:
        plot_performance_by_data_size(
            cfg_paths,
            parcellation_ids=parcellation_ids,
            reference_atlas_names=reference_atlas_names,
            evaluation_atlas_names=evaluation_atlas_names,
            plot_dir=join(output_dir, 'performance_by_data_size'),
            dump_data=dump_data
        )
    if plot_type & {'grid', 'all'}:
        plot_grid(
            cfg_paths,
            aggregation_ids=aggregation_ids,
            dimensions=dimensions,
            reference_atlas_names=reference_atlas_names,
            evaluation_atlas_names=evaluation_atlas_names,
            baseline_atlas_names=baseline_atlas_names,
            plot_dir=join(output_dir, 'grid'),
            dump_data=dump_data
        )

f.close()
