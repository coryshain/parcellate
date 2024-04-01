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
    if isinstance(cfg_paths, str):
        cfg_paths = [cfg_paths]

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
        if not os.path.exists(cfg_path):
            continue
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

    out = {}
    if not os.path.exists(cfg_path):
        return out

    cfg = get_cfg(cfg_path)
    output_dir = os.path.normpath(cfg['output_dir'])
    compressed = cfg.get('compress_outputs', True)
    suffix = get_suffix(compressed=compressed)

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
        include_thresholds=False,
        plot_dir=join('plots', 'performance'),
        reference_atlas_name_to_label=REFERENCE_ATLAS_NAME_TO_LABEL,
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
                df = pd.read_csv(df_path)
                df['cfg_path'] = cfg_path
                dfs[parcellation_id].append(df)

    for parcellation_id in dfs:
        if not (parcellation_id in dfs and len(dfs[parcellation_id])):
            continue
        df = pd.concat(dfs[parcellation_id], axis=0)
        atlas_names = df[df.parcel_type != 'baseline'].parcel.unique().tolist()
        _reference_atlas_names = df['atlas'].unique().tolist()
        if reference_atlas_names is None:
            _reference_atlas_names = df['atlas'].unique().tolist()
        else:
            _reference_atlas_names = [x for x in reference_atlas_names if x in _reference_atlas_names]

        for atlas_name in atlas_names:
            for reference_atlas_name in _reference_atlas_names:
                labels = [reference_atlas_name_to_label[reference_atlas_name], 'FC']

                # Similarity to reference
                _df = df[(df.parcel == atlas_name)]
                cols = ['atlas_score'] + ['jaccard%s' % x for x in SUFFIX2NAME]
                __df = _df[_df.atlas == reference_atlas_name][cols].rename(_rename_performance, axis=1)
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
                if dump_data:
                    __df.to_csv(
                        join(plot_dir, '%s_v_reference_%s_sim.csv' % (atlas_name, reference_atlas_name)),
                        index=False
                    )

                _evaluation_atlas_names = [x[:-6] for x in df if x.endswith('_score') and not x.startswith('atlas')]
                if evaluation_atlas_names is not None:
                    _evaluation_atlas_names = [x for x in evaluation_atlas_names if x in _evaluation_atlas_names]

                for evaluation_atlas_name in _evaluation_atlas_names:
                    suffixes = ['']
                    if include_thresholds:
                        suffixes += list(SUFFIX2NAME.keys())

                    # Similarity to evaluation
                    _dfr = df[df.parcel == 'reference_atlas_%s' % reference_atlas_name]
                    cols = ['%s_score%s' % (evaluation_atlas_name, s) for s in suffixes]
                    __df = _df[_df.atlas == reference_atlas_name][cols].rename(_rename_performance, axis=1)
                    __dfr = _dfr[_dfr.atlas == reference_atlas_name][cols].rename(_rename_performance, axis=1)
                    __df['label'] = labels[0]
                    __dfr['label'] = labels[1]
                    ylab = 'Similarity'
                    xlab = None
                    colors = ['c', 'm']
                    fig = _plot_performance(
                        __dfr, __df,
                        colors=colors,
                        labels=labels,
                        xlabel=xlab,
                        ylabel=ylab,
                        divider=True
                    )
                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    evaluation_name = '_'.join(_evaluation_atlas_names)
                    fig.savefig(join(plot_dir, '%s_v_evaluation_%s_sim.png' % (
                        atlas_name, evaluation_name)), dpi=300)
                    if dump_data:
                        csv = pd.concat([__df, __dfr], axis=0)
                        csv.to_csv(
                            join(plot_dir, '%s_v_evaluation_%s_sim.csv' % (atlas_name, reference_atlas_name)),
                            index=False
                        )

                    # Evaluation contrast size
                    __df = _df[_df.atlas == reference_atlas_name][cols].rename(_rename_performance, axis=1)
                    __dfr = _dfr[_dfr.atlas == reference_atlas_name][cols].rename(_rename_performance, axis=1)
                    __df['label'] = labels[0]
                    __dfr['label'] = labels[1]
                    ylab = '%s Contrast' % evaluation_atlas_name
                    xlab = None
                    colors = ['c', 'm']
                    fig = _plot_performance(
                        __dfr, __df,
                        colors=colors,
                        labels=labels,
                        xlabel=xlab,
                        ylabel=ylab,
                        divider=True
                    )
                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    fig.savefig(join(plot_dir, '%s_%s_contrast.png' % (
                        atlas_name, evaluation_atlas_name)), dpi=300)
                    if dump_data:
                        csv = pd.concat([__df, __dfr], axis=0)
                        csv.to_csv(
                            join(plot_dir, '%s_%s_contrast.csv' % (atlas_name, evaluation_atlas_name)),
                            index=False
                        )


def _plot_performance(
        *dfs,
        colors=None,
        labels=None,
        xlabel=None,
        ylabel=None,
        divider=False,
        width=None,
        height=3
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
        df['label'] = labels[i]
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


def _rename_performance(x):
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


def plot_grid(
        cfg_paths,
        aggregation_ids=None,
        dimensions=None,
        reference_atlas_names=None,
        evaluation_atlas_names=None,
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
            if os.path.exists(df_path):
                if aggregation_id not in dfs:
                    dfs[aggregation_id] = []
                df = pd.read_csv(df_path)
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
        _reference_atlas_names = df['atlas'].unique().tolist()
        if reference_atlas_names is None:
            _reference_atlas_names = df['atlas'].unique().tolist()
        else:
            _reference_atlas_names = [x for x in reference_atlas_names if x in _reference_atlas_names]

        for atlas_name in atlas_names:
            for reference_atlas_name in _reference_atlas_names:
                labels = ['FC', reference_atlas_name_to_label[reference_atlas_name]]

                for dimension in _dimensions:
                    _dimensions_other = [x for x in _dimensions if x != dimension]
                    # Similarity to reference
                    if dimension not in df:
                        df[dimension] = df.grid_id.apply(_get_param_value, args=(dimension,))
                    perf_col = 'atlas_score'
                    _df = df[(df.parcel == atlas_name)]
                    _df = _df[_df.atlas == reference_atlas_name]
                    selected = _df[_df.selected][[dimension, perf_col]]
                    selected = selected.set_index(dimension)[perf_col]
                    __df = _df.pivot(
                        columns=[dimension] + _dimensions_other,
                        index='cfg_path',
                        values=perf_col
                    )
                    __df.columns.name = _rename_grid(__df.columns.name)

                    fig = _plot_grid(
                        __df,
                        labels=labels,
                        selected=selected,
                        colors=['m'],
                        ylabel=_rename_grid(perf_col)
                    )

                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    fig.savefig(
                        join(plot_dir, '%s_v_reference_%s_sim.png' % (atlas_name, reference_atlas_name)),
                        dpi=300
                    )
                    if dump_data:
                        __df.to_csv(
                            join(plot_dir, '%s_v_reference_%s_sim.csv' % (atlas_name, reference_atlas_name)),
                            index=False
                        )

                    # Similarity to evaluation
                    _dfr = df[df.parcel == 'reference_atlas_%s' % reference_atlas_name]
                    _dfr = _dfr[_dfr.atlas == reference_atlas_name]
                    _evaluation_atlas_names = [x[:-6] for x in df if x.endswith('_score') and not x.startswith('atlas')]
                    if evaluation_atlas_names is not None:
                        _evaluation_atlas_names = [x for x in evaluation_atlas_names if x in _evaluation_atlas_names]
                    for evaluation_atlas_name in _evaluation_atlas_names:
                        perf_col = '%s_score' % evaluation_atlas_name
                        selected = _df[_df.selected][[dimension, perf_col]]
                        selected = selected.set_index(dimension)[perf_col]
                        __df = _df.pivot(
                            columns=[dimension] + _dimensions_other,
                            index='cfg_path',
                            values=perf_col
                        )
                        __df.columns.name = _rename_grid(__df.columns.name)
                        __dfr = _dfr.pivot(
                            columns=[dimension] + _dimensions_other,
                            index='cfg_path',
                            values=perf_col
                        )
                        __dfr.columns.name = _rename_grid(__dfr.columns.name)
                        __df['label'] = labels[0]
                        __dfr['label'] = labels[1]

                        fig = _plot_grid(
                            __df,
                            dfr=__dfr,
                            labels=labels,
                            selected=selected,
                            colors=['m', 'gray'],
                            ylabel=_rename_grid(perf_col)
                        )

                        if not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)
                        fig.savefig(
                            join(plot_dir, '%s_v_evaluation_%s_sim.png' % (atlas_name, evaluation_atlas_name)),
                            dpi=300
                        )
                        if dump_data:
                            csv = pd.concat([__df, __dfr], axis=0)
                            csv.to_csv(
                                join(plot_dir, '%s_v_evaluation_%s_sim.csv' % (atlas_name, evaluation_atlas_name)),
                                index=False
                            )

                        perf_col = '%s_contrast' % evaluation_atlas_name
                        selected = _df[_df.selected][[dimension, perf_col]]
                        selected = selected.set_index(dimension)[perf_col]
                        __df = _df.pivot(
                            columns=[dimension] + _dimensions_other,
                            index='cfg_path',
                            values=perf_col
                        )
                        __df.columns.name = _rename_grid(__df.columns.name)
                        __dfr = _dfr.pivot(
                            columns=[dimension] + _dimensions_other,
                            index='cfg_path',
                            values=perf_col
                        )
                        __dfr.columns.name = _rename_grid(__dfr.columns.name)
                        __df['label'] = labels[0]
                        __dfr['label'] = labels[1]

                        fig = _plot_grid(
                            __df,
                            dfr=__dfr,
                            labels=labels,
                            selected=selected,
                            colors=['m', 'gray'],
                            ylabel=_rename_grid(perf_col)
                        )

                        if not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)
                        fig.savefig(
                            join(plot_dir, '%s_%s_contrast.png' % (atlas_name, evaluation_atlas_name)),
                            dpi=300
                        )
                        if dump_data:
                            csv = pd.concat([__df, __dfr], axis=0)
                            csv.to_csv(
                                join(plot_dir, '%s_%s_contrast.csv' % (atlas_name, evaluation_atlas_name)),
                                index=False
                            )


def _plot_grid(
        df,
        dfr=None,
        labels=None,
        selected=None,
        colors=None,
        ylabel=None,
        width=4,
        height=3,
):
    plt.close('all')
    xlabel = df.columns.name

    dfs = [df]
    if dfr is not None:
        dfs.append(dfr)
    i = 0
    for i, _df in enumerate(dfs):
        label = labels[i]
        x = _df.columns
        y = _df.mean(axis=0)
        yerr = _df.sem(axis=0)
        if colors and i < len(colors):
            color = colors[i]
        else:
            color = None

        if len(_df) > 1:
            plt.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.2, linewidth=0, zorder=i, label=label)
        if i == 0:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'
        plt.plot(x, y, color=color, linestyle=linestyle, zorder=i, label=label)

    if selected is not None and len(selected):
        index = pd.Series(selected.index)
        x = index.mean()
        xerr = index.sem()
        if not np.isfinite(xerr):
            xerr = None
        y = selected.mean()
        yerr = selected.sem()
        if not np.isfinite(yerr):
            yerr = None
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='c,', linewidth=1, markersize=3, zorder=i+1)

    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

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
    except TypeError:
        pass

    return val


def _rename_grid(x):
    for suffix in SUFFIX2NAME:
        if x.endswith(suffix):
            return SUFFIX2NAME[suffix]
    if x == 'atlas_score':
        return 'Similarity to Reference'
    if x == 'n_networks':
        return 'N Networks'
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
    argparser.add_argument('-t', '--plot_type', nargs='+', default=['all'], help=textwrap.dedent('''\
        Type of plot to generate. One of ``atlas``, ``performance``, ``grid``, or ``all``.
    '''))
    argparser.add_argument('-p', '--parcellation_ids', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of parcellation(s) to use for plotting. If None, use all available parcellations.
    '''))
    argparser.add_argument('-a', '--aggregation_ids', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of aggregation(s) to use for plotting. If None, use all available parcellations.
    '''))
    argparser.add_argument('-r', '--reference_atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of reference atlas(es) to use for plotting. If None, use all available reference atlases.'''
    ))
    argparser.add_argument('-e', '--evaluation_atlas_names', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of evaluation atlas(es) to use for plotting. If None, use all available evaluation atlases.'''
    ))
    argparser.add_argument('-d', '--dimensions', nargs='+', default=None, help=textwrap.dedent('''\
        Name(s) of grid-searched dimension(s) to plot. If None, use all available dimensions.'''
    ))
    argparser.add_argument('-T', '--include_thresholds', action='store_true', help=textwrap.dedent('''\
        Include performance when binarizing the atlas at different probability thresholds.'''
    ))
    argparser.add_argument('-D', '--dump_data', action='store_true', help=textwrap.dedent('''\
        Save plot data to CSV.'''
    ))
    argparser.add_argument('-o', '--output_dir', default='plots', help=textwrap.dedent('''\
        Output directory for performance and grid plots (atlases are saved in each model directory).
    '''))
    args = argparser.parse_args()

    cfg_paths = args.cfg_paths
    plot_type = set(args.plot_type)
    parcellation_ids = args.parcellation_ids
    aggregation_ids = args.aggregation_ids
    reference_atlase_names = args.reference_atlas_names
    evaluation_atlase_names = args.evaluation_atlas_names
    dimensions = args.dimensions
    include_thresholds = args.include_thresholds
    dump_data = args.dump_data
    output_dir = args.output_dir

    if plot_type & {'atlas', 'all'}:
        plot_atlases(
            cfg_paths,
            parcellation_ids,
            reference_atlase_names,
            evaluation_atlase_names
        )
    if plot_type  & {'performance', 'all'}:
        plot_performance(
            cfg_paths,
            parcellation_ids=parcellation_ids,
            reference_atlas_names=reference_atlase_names,
            evaluation_atlas_names=evaluation_atlase_names,
            include_thresholds=include_thresholds,
            plot_dir=join(output_dir, 'performance'),
            dump_data=dump_data
        )
    if plot_type & {'grid', 'all'}:
        plot_grid(
            cfg_paths,
            aggregation_ids=aggregation_ids,
            dimensions=dimensions,
            reference_atlas_names=reference_atlase_names,
            evaluation_atlas_names=evaluation_atlase_names,
            plot_dir=join(output_dir, 'grid'),
            dump_data=dump_data
        )