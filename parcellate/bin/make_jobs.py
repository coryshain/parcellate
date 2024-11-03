import sys
import os
import argparse

from parcellate.cfg import get_cfg
from parcellate.util import get_path


base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=%d:00:00
#SBATCH --mem=%dgb
#SBATCH --ntasks=%d
"""

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run parcellations specified in one or more config (YAML) files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s).')
    argparser.add_argument('-p', '--partition', nargs='+', help='Partition(s) over which to predict/evaluate')
    argparser.add_argument('-t', '--time', type=int, default=24, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n_cores', type=int, default=1, help='Number of cores to request')
    argparser.add_argument('-m', '--memory', type=int, default=8, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--slurm_partition', default=None, help='Value for SLURM --partition setting, if applicable')
    argparser.add_argument('-e', '--exclude', nargs='+', help='Nodes to exclude')
    argparser.add_argument('-s', '--storage_dir', help='Path to directory to move results to upon completion of the job. A softlink will be created at the original path')
    argparser.add_argument('-d', '--n_subdir', type=int, default=1, help=('Number of additional parent subdirectories to include in '
                                                                  'end-training move operation. Ignored unless --storage_dir is used.'))
    argparser.add_argument('-g', '--grid_only', action='store_true', help='Only fit grid search models, do not aggregate or refit')
    argparser.add_argument('-o', '--outdir', default='./', help='Directory in which to place generated batch scripts')
    args = argparser.parse_args()

    paths = args.paths
    partitions = args.partition
    time = args.time
    n_cores = args.n_cores
    memory = args.memory
    slurm_partition = args.slurm_partition
    if args.exclude:
        exclude = ','.join(args.exclude)
    else:
        exclude = []
    storage_dir = args.storage_dir
    n_subdir = args.n_subdir
    grid_only = args.grid_only
    if grid_only:
        grid_only = ' -g'
    else:
        grid_only = ''
    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for path in paths:
        job_name = os.path.basename(path)[:-4] + '_parcellate'
        filename = outdir + '/' + job_name + '.pbs'
        with open(filename, 'w') as f:
            f.write(base % (job_name, job_name, time, memory, n_cores))
            if slurm_partition:
                f.write('#SBATCH --partition=%s\n' % slurm_partition)
            if exclude:
                f.write('#SBATCH --exclude=%s\n' % exclude)
            f.write('\n\nset -e\n\n')
            f.write('python -m parcellate.bin.train %s%s -P\n' % (path, grid_only))
            if storage_dir:
                softlink_dir = get_cfg(path)['output_dir']
                data_dir = os.path.basename(softlink_dir)
                parent_dir = os.path.dirname(softlink_dir)
                for _ in range(n_subdir):
                    data_dir = os.path.join(os.path.basename(parent_dir), data_dir)
                    parent_dir = os.path.dirname(data_dir)
                data_dir = os.path.join(os.path.normpath(os.path.realpath(storage_dir)), data_dir)
                f.write('if ! [[ -L %s ]]; then\n' % softlink_dir)  # Stop if the target is already a softlink
                f.write('    echo "Ensuring directory %s exists"\n' % (os.path.dirname(data_dir)))
                f.write('    mkdir -p %s\n' % (os.path.dirname(data_dir)))
                f.write('    if [[ -d %s ]]; then\n' % softlink_dir)  # Stop if the target doesn't exist
                f.write('        if [[ -d %s ]]; then\n' % data_dir)  # Stop if the target doesn't exist
                f.write('            echo "Removing stale target %s"\n' % data_dir)
                f.write('            rm -r %s\n' % data_dir)  # Stop if the target doesn't exist
                f.write('        fi\n')
                f.write('        echo "Moving directory %s to storage"\n' % softlink_dir)
                f.write('        mv %s %s\n' % (softlink_dir, os.path.dirname(data_dir)))
                f.write('    fi\n')
                f.write('    echo "Linking to directory %s"\n' % data_dir)
                f.write('    ln -s %s %s\n' % (data_dir, softlink_dir))
                f.write('fi\n')

