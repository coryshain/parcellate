import re

CFG_FILENAME = 'config.yml'
DEFAULT_ID = 'main'
N_INIT = 1
INIT_SIZE = None
TRAILING_DIGITS = re.compile('.*?([0-9]*)$')
ACTION_VERB_TO_NOUN = dict(
    sample='sample',
    align='alignment',
    evaluate='evaluation',
    aggregate='aggregation',
    parcellate='parcellation',
)
REFERENCE_ATLAS_PREFIX = 'reference_atlas_'
EVALUATION_ATLAS_PREFIX = 'evaluation_atlas_'
PATHS = dict(
    sample=dict(
        kwargs='sample_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['sample'],
        output='sample%s',
    ),
    align=dict(
        kwargs='align_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['align'],
        output='parcellation%s',
        evaluation='evaluation.csv'
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
        output='parcellation%s',
        evaluation='evaluation.csv'
    ),
    grid=dict(
        subdir='grid'
    )
)