import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict, OrderedDict 
try:
    import wandb
except Exception as e:
    pass
from tqdm import tqdm
import numpy as np
import numpy.random as npr
import torch.nn.functional as F

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset
from wilds.datasets.counterfactualtext_dataset import Counterfactual, CounterfactualTextDataset

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to
from train import train, evaluate, infer_predictions
from algorithms.initializer import initialize_algorithm, infer_d_out
from transforms import initialize_transform
from models.initializer import initialize_model
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing

# Necessary for large images of GlobalWheat
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification

class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ) 
        return outputs

class BertClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.num_labels
        
    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ) 
        return outputs
    
def load_bert_based_model(config, d_out): 
    config.model_kwargs['output_hidden_states'] = True
    if config.model == 'bert-base-uncased':
        model = BertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)

    elif config.model == 'distilbert-base-uncased':
        model = DistilBertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    else:
        raise ValueError(
            "{} is not supported. Choices are 'distilbert-base-uncased' and 'bert-base-uncased'".format(config.model))
        
    if config.model_kwargs['state_dict'] is not None:
        print(f'Initialized model with pretrained weights from {config.pretrained_model_path}')
        model.load_state_dict(config.model_kwargs['state_dict'] )
    else:
        raise ValueError("config.model_kwargs['state_dict'] is not specified.")
    return model

def main():
    
    ''' Arg defaults are filled in according to examples/configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--noise_ratio', type=float, default=0.2, help='Max proportion to flip the labels')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # NoisyStudent-specific loading
    parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--use_pth', type=str, default="best")
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    config = parser.parse_args()
    config = populate_defaults(config)

    # For the GlobalWheat detection dataset,
    # we need to change the multiprocessing strategy or there will be
    # too many open file descriptors.
    if config.dataset == 'globalwheat':
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")

    # Set random seed
    set_seed(config.seed)

    # Data
    if "noisy" in config.dataset:
        full_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.version,
            root_dir=config.root_dir,
            download=config.download,
            split_scheme=config.split_scheme,
            noise_ratio=config.noise_ratio,
            seed=config.seed,
            **config.dataset_kwargs)
    else:
        full_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.version,
            root_dir=config.root_dir,
            download=config.download,
            split_scheme=config.split_scheme,
            **config.dataset_kwargs)

    # Transforms & data augmentations for labeled dataset
    # To modify data augmentation, modify the following code block.
    # If you want to use transforms that modify both `x` and `y`,
    # set `do_transform_y` to True when initializing the `WILDSSubset` below.
    train_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        additional_transform_name=config.additional_train_transform,
        is_training=True)
    eval_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=False)

    # Configure unlabeled datasets
    unlabeled_dataset = None
    if config.unlabeled_split is not None:
        split = config.unlabeled_split
        full_unlabeled_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.unlabeled_version,
            root_dir=config.root_dir,
            download=config.download,
            unlabeled=True,
            **config.dataset_kwargs
        )
        train_grouper = CombinatorialGrouper(
            dataset=[full_dataset, full_unlabeled_dataset],
            groupby_fields=config.groupby_fields
        )

        # Transforms & data augmentations for unlabeled dataset
        if config.algorithm == "FixMatch":
            # For FixMatch, we need our loader to return batches in the form ((x_weak, x_strong), m)
            # We do this by initializing a special transform function
            unlabeled_train_transform = initialize_transform(
                config.transform, config, full_dataset, is_training=True, additional_transform_name="fixmatch"
            )
        else:
            # Otherwise, use the same data augmentations as the labeled data.
            unlabeled_train_transform = train_transform

        if config.algorithm == "NoisyStudent":
            # For Noisy Student, we need to first generate pseudolabels using the teacher
            # and then prep the unlabeled dataset to return these pseudolabels in __getitem__
            print("Inferring teacher pseudolabels for Noisy Student")
            assert config.teacher_model_path is not None
            if not config.teacher_model_path.endswith(".pth"):
                # Use the best model
                config.teacher_model_path = os.path.join(
                    config.teacher_model_path,  f"{config.dataset}_seed:{config.seed}_epoch:best_model.pth"
                )

            d_out = infer_d_out(full_dataset, config)
            teacher_model = initialize_model(config, d_out).to(config.device)
            load(teacher_model, config.teacher_model_path, device=config.device)
            # Infer teacher outputs on weakly augmented unlabeled examples in sequential order
            weak_transform = initialize_transform(
                transform_name=config.transform,
                config=config,
                dataset=full_dataset,
                is_training=True,
                additional_transform_name="weak"
            )
            unlabeled_split_dataset = full_unlabeled_dataset.get_subset(split, transform=weak_transform, frac=config.frac)
            sequential_loader = get_eval_loader(
                loader=config.eval_loader,
                dataset=unlabeled_split_dataset,
                grouper=train_grouper,
                batch_size=config.unlabeled_batch_size,
                **config.unlabeled_loader_kwargs
            )
            teacher_outputs = infer_predictions(teacher_model, sequential_loader, config)
            teacher_outputs = move_to(teacher_outputs, torch.device("cpu"))
            unlabeled_split_dataset = WILDSPseudolabeledSubset(
                reference_subset=unlabeled_split_dataset,
                pseudolabels=teacher_outputs,
                transform=unlabeled_train_transform,
                collate=full_dataset.collate,
            )
            teacher_model = teacher_model.to(torch.device("cpu"))
            del teacher_model
        else:
            unlabeled_split_dataset = full_unlabeled_dataset.get_subset(
                split, 
                transform=unlabeled_train_transform, 
                frac=config.frac, 
                load_y=config.use_unlabeled_y
            )

        unlabeled_dataset = {
            'split': split,
            'name': full_unlabeled_dataset.split_names[split],
            'dataset': unlabeled_split_dataset
        }
        unlabeled_dataset['loader'] = get_train_loader(
            loader=config.train_loader,
            dataset=unlabeled_dataset['dataset'],
            batch_size=config.unlabeled_batch_size,
            uniform_over_groups=config.uniform_over_groups,
            grouper=train_grouper,
            distinct_groups=config.distinct_groups,
            n_groups_per_batch=config.unlabeled_n_groups_per_batch,
            **config.unlabeled_loader_kwargs
        )
    else:
        train_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=config.groupby_fields
        )
    if config.unlabeled_split is None and unlabeled_dataset is None:
        print("Not using unlabed dataset")
    # Configure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

    if config.use_wandb:
        initialize_wandb(config)
 
    train_dataset = datasets['train']['dataset'] 
    d_out = infer_d_out(train_dataset, config) 
    assert full_dataset.n_classes == d_out
    
    # Load the pretrained weights
    if "noisy" in config.dataset:
        filename = "{}_seed:{}_epoch:{}_model.pth".format(config.dataset.split("_")[0], config.seed, config.use_pth)
    else:
        filename = "{}_seed:{}_epoch:{}_model.pth".format(config.dataset, config.seed, config.use_pth)
    pretrained_model_path = os.path.join(config.log_dir, filename)
    pretrained_model_weights = torch.load(pretrained_model_path, map_location=config.device)['algorithm']
    pretrained_model_weights = OrderedDict([(k[6:], v) for k, v in pretrained_model_weights.items()])
    if config.model in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50', 'densenet121'):
        # Initialize algorithm 
        model = initialize_model(config, d_out=d_out)   
        model.load_state_dict(pretrained_model_weights)
    elif 'bert' in config.model: 
        config.model_kwargs['state_dict'] = pretrained_model_weights
        model = load_bert_based_model(config, d_out)
    else:
        raise ValueError(
            "{} is not supported. Current choices include bert-based or 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50', 'densenet121'.".format(config.model))
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
 
    if config.model in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50', 'densenet121'):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        if config.model == "densenet121": 
            model.features.register_forward_hook(get_activation('features'))
        else:  
            model.avgpool.register_forward_hook(get_activation('avgpool'))
    
    if 'id_val' in list(full_dataset.split_dict.keys()) or 'id_test' in list(full_dataset.split_dict.keys()):
        savedict = {
            'train':'train', 'id_val':'valid', 'id_test':'test',
            'val':'valid_ood', 'test':'test_ood'
        } 
    else:
        savedict = {
            'train':'train', 'val':'valid', 'test':'test',
            'val_ood':'valid_ood', 'test_ood':'test_ood'
        } 
        
 
    for phase in list(full_dataset.split_dict.keys()):
        file_path = os.path.join(config.log_dir, '{}_{}.npz'.format(
            savedict[phase], config.use_pth
        ))
        if not os.path.isfile(file_path):
            # Iterate over data.
            results = {'labels': [], 'labels_orc': [], 'logits': [], 'features': []}
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for minibatch in tqdm(datasets[phase]['loader'], desc=phase):
                if len(minibatch) == 3:
                    inputs, labels, _ = minibatch 
                    results['labels'].append(labels.numpy())
                    results['labels_orc'].append(labels.numpy())
                else:
                    inputs, labels, _, labels_orc = minibatch
                    results['labels'].append(labels.numpy())
                    results['labels_orc'].append(labels_orc.numpy())
                inputs = inputs.to(device) 
                if config.model == 'distilbert-base-uncased':
                    outputs = model(inputs) 
                    pooled_output = outputs['hidden_states'][-1][:, 0]  # (bs, dim)
                    pooled_output = model.pre_classifier(pooled_output)  # (bs, dim)
                    features = nn.ReLU()(pooled_output)  # (bs, dim) 
                    # CHECK: outputs['logits'] == model.classifier(features) # (bs, num_labels) 
                    results['logits'].append(outputs['logits'].cpu().detach().numpy())
                    results['features'].append(features.cpu().detach().numpy())
                elif config.model == 'bert-base-uncased':
                    outputs = model(inputs) 
                    features = model.bert.pooler(outputs['hidden_states'][-1])  # (bs, dim)
                    # CHECK: outputs['logits'] == model.classifier(features) # (bs, num_labels)
                    results['features'].append(features.cpu().detach().numpy())
                    results['logits'].append(outputs['logits'].cpu().detach().numpy())
                elif config.model in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50'):
                    logits = model(inputs)  
                    results['logits'].append(logits.cpu().detach().numpy())
                    results['features'].append(activation['avgpool'].cpu().numpy()[:, :, 0, 0])
                elif config.model == 'densenet121':
                    logits = model(inputs)  
                    results['logits'].append(logits.cpu().detach().numpy())
                    out = F.relu(activation['features'], inplace=True)
                    out = F.adaptive_avg_pool2d(out, (1, 1))
                    results['features'].append(torch.flatten(out, 1).cpu().numpy())
            for k in results:
                results[k] = np.concatenate(results[k], axis=0)
                
            np.savez(file_path, **results)

if __name__=='__main__':
    main() 
