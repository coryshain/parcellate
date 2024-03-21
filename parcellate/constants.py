import re

N_INIT = 1
INIT_SIZE = None
CFG_FILENAME = 'config.yml'
SAMPLE_FILENAME_BASE = 'sample'
PARCELLATE_CFG_FILENAME = 'parcellate.yml'
PARCELLATION_SUBDIR = 'parcellation'
PARCELLATION_FILENAME_BASE = 'parcellation'
ALIGN_CFG_FILENAME = 'align.yml'
ALIGNMENT_SUBDIR = 'alignment'
ALIGNMENT_FILENAME_BASE = 'parcellation'
ALIGNMENT_EVALUATION_FILENAME = 'evaluation.csv'
EVALUATE_CFG_FILENAME = 'evaluate.yml'
EVALUATION_SUBDIR = 'evaluation'
EVALUATION_FILENAME = 'evaluation.csv'
AGGREGATE_CFG_FILENAME = 'aggregate.yml'
AGGREGATION_SUBDIR = 'aggregation'
AGGREGATION_FILENAME = 'aggregation.yml'
AGGREGATION_EVALUATION_FILENAME = 'evaluation.csv'
GRID_SUBDIR = 'grid'
GRID_CFG_FILENAME = 'grid.yml'
FINAL_PARCELLATION_SUBDIR = 'parcellation'
TRAILING_DIGITS = re.compile('.*?([0-9]*)$')