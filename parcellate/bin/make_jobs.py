import sys
import os
import argparse

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=%d:00:00
#SBATCH --mem=%dgb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%d
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
            f.write('\n')
            f.write('python -m parcellate.bin.main %s -P\n' % path)
