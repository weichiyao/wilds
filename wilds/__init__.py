from .version import __version__
from .get_dataset import get_dataset

benchmark_datasets = [
    'amazon',
    'camelyon17', 
    'civilcomments',
    'iwildcam', 
    'ogb-molpcba',
    'poverty',
    'fmow',
    'py150',
    'rxrx1', 
    'globalwheat',
]

noisy_benchmark_datasets = [
    'camelyon17_noisy',
    'iwildcam_noisy',
    'rxrx1_noisy',
]

additional_datasets = [
    'celebA',
    'domainnet',
    'waterbirds',
    'yelp',
    'bdd100k',
    'sqf',
    'encode'
]

counterfactual_text_datasets = [
    'kindle', 
    'imdb', 
    'imdb_sents'
]

retrain_datasets = [
    'retrain'   
]

supported_datasets = benchmark_datasets + additional_datasets + noisy_benchmark_datasets + counterfactual_text_datasets + retrain_datasets

unlabeled_datasets = [
    'amazon',
    'camelyon17',
    'domainnet',
    'civilcomments',
    'iwildcam',
    'ogb-molpcba',
    'poverty',
    'fmow',
    'globalwheat',
]

unlabeled_splits = [
    'train_unlabeled',
    'val_unlabeled',
    'test_unlabeled',
    'extra_unlabeled'
]
