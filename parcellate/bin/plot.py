import os
import subprocess
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from PIL import Image
import textwrap
import pprint
from nilearn import image
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
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
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


######################################
#
#  ATLAS
#
######################################


def plot_atlases(
        cfg_paths,
        parcellation_ids=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None
):
    binary_dir = join(dirname(dirname(dirname(__file__))), 'resources', 'surfice', 'Surf_Ice')
    assert os.path.exists(binary_dir), ('Surf Ice directory %s not found. Install using '
        '``python -m parcellate.bin.install_surf_ice``.' % binary_dir)
    binary_path = None
    for path in os.listdir(binary_dir):
        if path in ('surfice', 'surfice.exe'):
            binary_path = join(binary_dir, path)
            break
    assert binary_path, 'No Surf Ice executable found'

    script = _get_surf_ice_script(
        cfg_paths,
        parcellation_ids,
        reference_atlas_names,
        evaluation_atlas_names=evaluation_atlas_names
    )

    subprocess.call([binary_path, '-S', script])

    for cfg_path in cfg_paths:
        cfg = get_cfg(cfg_path)
        output_dir = cfg['output_dir']
        for parcellation_dir in os.listdir(join(output_dir, 'parcellation')):
            if parcellation_ids is None or \
                    parcellation_dir in parcellation_ids or \
                    parcellation_dir == parcellation_ids:
                parcellation_dir = join(output_dir, 'parcellation', parcellation_dir, 'plots')
                img_prefixes = set()
                for img in [x for x in os.listdir(parcellation_dir) if _is_hemi(x)]:
                    img_prefix = '_'.join(img.split('_')[:-2])
                    img_prefix = join(parcellation_dir, img_prefix)
                    img_prefixes.add(img_prefix)
                for img_prefix in img_prefixes:
                    imgs = []
                    for hemi in ('left', 'right'):
                        if hemi == 'left':
                            views = ('lateral', 'medial')
                        else:
                            views = ('medial', 'lateral')
                        for view in views:
                            imgs.append(Image.open(img_prefix + '_%s_%s.png' % (hemi, view)))
                    widths, heights = zip(*(i.size for i in imgs))
                    total_width = sum(widths)
                    max_height = max(heights)
                    new_im = Image.new('RGB', (total_width, max_height))
                    x_offset = 0
                    for im in imgs:
                        new_im.paste(im, (x_offset, 0))
                        x_offset += im.size[0]
                    new_im.save('%s.png' % img_prefix)


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

    cfg = get_cfg(cfg_path)
    output_dir = os.path.normpath(cfg['output_dir'])
    compressed = cfg.get('compress_outputs', True)
    suffix = get_suffix(compressed=compressed)

    out = {}
    if parcellation_ids is None:
        parcellation_ids = os.listdir(join(output_dir, 'parcellation'))
    for parcellation_id in parcellation_ids:
        parcellation_dir = get_path(output_dir, 'subdir', 'parcellate', parcellation_id, compressed=compressed)
        if os.path.exists(parcellation_dir):
            out[parcellation_id] = dict(
                reference_atlases={},
                evaluation_atlases={},
                atlases={}
            )
            for filename in [x for x in os.listdir(parcellation_dir) if x.endswith(suffix)]:
                filepath = join(parcellation_dir, filename)
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

    return out


def _get_surf_ice_script(
        cfg_paths,
        parcellation_ids=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None
):
    script = textwrap.dedent('''\
    import sys
    import os
    import gl

    CWD = os.path.normpath(os.path.join('..', '..', '..', os.getcwd()))

    MIN = dict(
        reference=0.3,
        evaluation=0.2,
        atlas=0.3
    )

    MAX = dict(
        reference=0.5,
        evaluation=1,
        atlas=0.5
    )

    IX = dict(
        reference=1,
        evaluation=1,
        atlas=2
    )

    COLOR = dict(
        reference=(0, 128, 0, 0, 255, 0),  # Green
        evaluation=(0, 0, 128, 0, 0, 255), # Blue
        atlas=(128, 0, 0, 255, 0, 0),      # Red
    )

    X = 400
    Y = 300

    plot_sets = [
    ''')

    for cfg_path in cfg_paths:
        atlas_paths = _get_atlas_paths(
            cfg_path,
            parcellation_ids=parcellation_ids,
            reference_atlas_names=reference_atlas_names,
            evaluation_atlas_names=evaluation_atlas_names
        )

        for parcellation_id in atlas_paths:
            for atlas_name in atlas_paths[parcellation_id]['atlases']:
                atlas_path = atlas_paths[parcellation_id]['atlases'][atlas_name]
                for reference_atlas_name in atlas_paths[parcellation_id]['reference_atlases']:
                    reference_atlas_path = atlas_paths[parcellation_id]['reference_atlases'][reference_atlas_name]
                    output_dir = dirname(atlas_path)
                    output_path = join(output_dir, 'plots', '%s_vs_%s_atlas_%s_%%s_%%s.png' % (
                        atlas_name, 'reference', reference_atlas_name))
                    plot_set = dict(
                        atlas=dict(
                            name=atlas_name,
                            path=atlas_path,
                            output_path=output_path,
                        ),
                        reference=dict(
                            name=reference_atlas_name,
                            path=reference_atlas_path
                        )
                    )
                    script += '    %s,\n' % pprint.pformat(plot_set)

                for evaluation_atlas_name in atlas_paths[parcellation_id]['evaluation_atlases']:
                    evaluation_atlas_path = atlas_paths[parcellation_id]['evaluation_atlases'][evaluation_atlas_name]
                    output_dir = dirname(atlas_path)
                    output_path = join(output_dir, 'plots', '%s_vs_%s_atlas_%s_%%s_%%s.png' % (
                        atlas_name, 'evaluation', evaluation_atlas_name))
                    plot_set = dict(
                        atlas=dict(
                            name=atlas_name,
                            path=atlas_path,
                            output_path=output_path,
                        ),
                        evaluation=dict(
                            name=evaluation_atlas_name,
                            path=evaluation_atlas_path
                        )
                    )
                    script += '    %s,\n' % pprint.pformat(plot_set)
    script += ']\n'

    script += textwrap.dedent('''\


    def get_path(path):
        if not os.path.isabs(path):
            path = os.path.join(CWD, os.path.normpath(path))
        path = os.path.normpath(path)

        return path

    for plot_set in plot_sets:
        atlas = plot_set['atlas']
        output_path = get_path(plot_set['atlas']['output_path'])
        comparison = plot_set.get('reference', None)
        if comparison is None:
            comparison = plot_set.get('evaluation', None)
            comparison_type = 'evaluation'
        else:
            comparison_type = 'reference'

        comparison_name = comparison['name']
        comparison_path = get_path(comparison['path'])
        atlas_name = atlas['name']
        atlas_path = get_path(atlas['path'])

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

                gl.overlayload(comparison_path)
                gl.overlaycolor(IX[comparison_type], *COLOR[comparison_type])
                gl.overlayminmax(IX[comparison_type], MIN[comparison_type], MAX[comparison_type])
                gl.overlayload(atlas_path)
                gl.overlaycolor(IX['atlas'], *COLOR['atlas'])
                gl.overlayminmax(IX['atlas'], MIN['atlas'], MAX['atlas'])
                gl.overlayadditive(1)
                gl.colorbarvisible(0)
                gl.orientcubevisible(0)
                gl.cameradistance(0.55)

                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                plot_path = output_path % (hemi, view)
                gl.savebmpxy(plot_path, X, Y)
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
#  PERFORMANCE
#
######################################


def plot_performance(
        cfg_paths,
        parcellation_ids=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
        plot_dir=join('plots', 'performance')
):
    if isinstance(parcellation_ids, str):
        parcellation_ids = [parcellation_ids]

    dfs = {}
    for cfg_path in cfg_paths:
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
        df = pd.concat(dfs[parcellation_id], axis=0)
        atlas_names = df[df.parcel_type != 'baseline'].parcel.unique().tolist()
        _reference_atlas_names = df['atlas'].unique().tolist()
        if reference_atlas_names is None:
            _reference_atlas_names = df['atlas'].unique().tolist()
        else:
            _reference_atlas_names = [x for x in reference_atlas_names if x in _reference_atlas_names]

        for atlas_name in atlas_names:
            for reference_atlas_name in _reference_atlas_names:
                # Similarity to reference
                _df = df[(df.parcel == atlas_name)]
                cols = ['atlas_score'] + ['jaccard%s' % x for x in SUFFIX2NAME]
                __df = _df[_df.atlas == reference_atlas_name][cols].rename(_rename, axis=1)
                colors = ['m']
                xlab = None
                ylab = 'Similarity'
                fig = _plot_performance(
                    __df,
                    colors=colors,
                    xlabel=xlab,
                    ylabel=ylab,
                    divider=True
                )
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig.savefig(join(plot_dir, '%s_v_reference_%s_sim.png' % (atlas_name, reference_atlas_name)), dpi=300)

                # Similarity to evaluation
                _evaluation_atlas_names = [x[:-6] for x in df if x.endswith('_score') and not x.startswith('atlas')]
                if evaluation_atlas_names is not None:
                    _evaluation_atlas_names = [x for x in evaluation_atlas_names if x in _evaluation_atlas_names]
                _dfr = df[df.parcel == 'reference_atlas_%s' % reference_atlas_name]
                cols = []
                for evaluation_atlas_name in _evaluation_atlas_names:
                    cols.append('%s_score' % evaluation_atlas_name)
                __df = _df[_df.atlas == reference_atlas_name][cols].rename(_rename, axis=1)
                __dfr = _dfr[_dfr.atlas == reference_atlas_name][cols].rename(_rename, axis=1)
                ylab = 'Similarity'
                xlab = None
                colors = ['c', 'm']
                fig = _plot_performance(
                    __dfr, __df,
                    colors=colors,
                    labels=['LanA', 'FC'],
                    xlabel=xlab,
                    ylabel=ylab,
                    divider=True
                )
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig.savefig(join(plot_dir, '%s_v_evaluation_%s_sim.png' % (
                    atlas_name, reference_atlas_name)), dpi=300)

                # Evaluation contrast size
                for evaluation_atlas_name in _evaluation_atlas_names:
                    cols = ['%s_contrast%s' % (evaluation_atlas_name, s) for s in [''] + list(SUFFIX2NAME.keys())]
                    __df = _df[_df.atlas == reference_atlas_name][cols].rename(_rename, axis=1)
                    __dfr = _dfr[_dfr.atlas == reference_atlas_name][cols].rename(_rename, axis=1)
                    ylab = '%s contrast (PSC)' % evaluation_atlas_name
                    xlab = None
                    colors = ['c', 'm']
                    fig = _plot_performance(
                        __dfr, __df,
                        colors=colors,
                        labels=['LanA', 'FC'],
                        xlabel=xlab,
                        ylabel=ylab,
                        divider=True
                    )
                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    fig.savefig(join(plot_dir, '%s_%s_contrast.png' % (
                        atlas_name, evaluation_atlas_name)), dpi=300)




def _plot_performance(
        *dfs,
        colors=None,
        labels=None,
        xlabel=None,
        ylabel=None,
        divider=False,
        width=None,
        height=3,
):
    plt.close('all')
    n_colors = len(dfs)
    bar_width = 0.8 / n_colors
    n_ticks = None
    tick_labels = None
    x = None
    xlim = None
    spacer = 1
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
        if colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = None
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


def _rename(x):
    for suffix in SUFFIX2NAME:
        if x.endswith(suffix):
            return SUFFIX2NAME[suffix]
    if x == 'atlas_score':
        return 'Overall'
    if x.endswith('_score'):
        return x[:-6]
    if x.endswith('_contrast'):
        return x[:-9]
    return x










######################################
#
#  GRID
#
######################################


def plot_grids(
        cfg_paths,
        dimensions=None,
        reference_atlases=None,
        evaluation_atlases=None,
        aggregation_id='main',
        output_dir='parcellate_plots'
):
    # TODO
    if dimensions is None:
        dimensions = []
    if reference_atlases is None:
        reference_atlases = []
    if evaluation_atlases is None:
        evaluation_atlases = {}
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]
    for cfg_path in cfg_paths:
        cfg = get_cfg(cfg_path)
        output_dir = cfg['output_dir']
        compressed = cfg.get('compress_outputs', True)
        df_path = get_aggregation_path(output_dir, aggregation_id=aggregation_id)
        df = pd.read_csv(df_path)
        print(df)


def plot_grid(
        cfg_paths,
        dimension,
        reference_atlas,
        evaluation_atlas,
        aggregation_id,
        output_dir=join('plots', 'performance')
):
    # TODO
    ...










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
    argparser.add_argument('-t', '--plot_type', default='all', help=textwrap.dedent('''\
        Type of plot to generate. One of ``atlas``, ``performance``, ``grid``, or ``all``.
    '''))
    argparser.add_argument('-p', '--parcellation_ids', nargs='+', default=None, help=textwrap.dedent('''\
        ID(s) of parcellation to use for plotting. If None, use all available parcellations.
    '''))
    argparser.add_argument('-r', '--reference_atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name of reference atlas(es) to use for plotting. If None, use all available reference atlases.'''
    ))
    argparser.add_argument('-e', '--evaluation_atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name of evaluation atlas(es) to use for plotting. If None, use all available evaluation atlases.'''
    ))
    argparser.add_argument('-d', '--dimensions', nargs='+', default=None, help=textwrap.dedent('''\
        Name of grid-searched dimension(s) to plot. If None, use all available dimensions.'''
    ))
    args = argparser.parse_args()

    cfg_paths = args.cfg_paths
    plot_type = args.plot_type
    parcellation_ids = args.parcellation_ids
    reference_atlase_names = args.reference_atlas_names
    evaluation_atlase_names = args.evaluation_atlas_names
    dimensions = args.dimensions

    if plot_type in ('atlas', 'all'):
        plot_atlases(
            cfg_paths,
            parcellation_ids,
            reference_atlase_names,
            evaluation_atlase_names
        )
    if plot_type in ('performance', 'all'):
        plot_performance(
            cfg_paths,
            parcellation_ids=parcellation_ids,
            reference_atlas_names=reference_atlase_names,
            evaluation_atlas_names=evaluation_atlase_names
        )
    if plot_type in ('grid', 'all'):
        plot_grids(
            cfg_paths,
            dimensions=dimensions,
            reference_atlases=reference_atlase_names,
            evaluation_atlases=evaluation_atlase_names,
            output_dir='parcellate_plots'
        )