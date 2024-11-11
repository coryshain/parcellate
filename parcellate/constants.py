import re
import numpy as np

CFG_FILENAME = 'config.yml'
DEFAULT_ID = 'main'
N_INIT = 1
INIT_SIZE = None
TRAILING_DIGITS = re.compile('.*?([0-9]*)$')
ACTION_VERB_TO_NOUN = dict(
    sample='sample',
    align='alignment',
    label='labeling',
    evaluate='evaluation',
    aggregate='aggregation',
    parcellate='parcellation',
)
REFERENCE_ATLAS_PREFIX = 'ref_'
EVALUATION_ATLAS_PREFIX = 'eval_'
ALL_REFERENCE = [
    'LANG',
    'FPN_A',
    'FPN_B',
    'DN_A',
    'DN_B',
    'CG_OP',
    'SAL_PMN',
    'dATN_A',
    'dATN_B',
    'AUD',
    'PM_PPr',
    'SMOT_A',
    'SMOT_B',
    'VIS_C',
    'VIS_P',
    'LANA',
]
PATHS = dict(
    sample=dict(
        kwargs='sample_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['sample'],
        output='sample%s',
        evaluation='evaluation.csv',
        metadata='metadata.csv',
    ),
    align=dict(
        kwargs='align_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['align'],
        output='parcellation%s',
        evaluation='evaluation.csv'
    ),
    label=dict(
        kwargs='label_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['label'],
        output='evaluation.csv',
    ),
    evaluate=dict(
        kwargs='evaluate_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['evaluate'],
        output='evaluation.csv',
    ),
    aggregate=dict(
        kwargs='aggregate_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['aggregate'],
        output='parcellate_kwargs.yml',
        evaluation='evaluation.csv'
    ),
    parcellate=dict(
        kwargs='parcellate_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['parcellate'],
        output='finished.txt',
        evaluation='evaluation.csv'
    ),
    grid=dict(
        subdir='grid'
    )
)
REFERENCE_ATLAS_NAME_TO_LABEL = dict()