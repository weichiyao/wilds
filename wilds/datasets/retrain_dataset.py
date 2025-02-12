import os
import time

import torch
import numpy as np
 
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms 
import datetime 
from tqdm import tqdm 
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper

class RetrainDataset:   
    DEFAULT_SPLIT_NAMES = {
        'train': 'Train',
        'val': 'Validation (ID)',
        'val_ood': 'Validation (OOD)',
        'test': 'Validation (ID)',
        'test_ood': 'Test (OOD)',
    }
    
    DEFAULT_SPLITS = {
        'train': 0, 
        'val': 1,
        'val_ood': 2,
        'test': 3,
        'test_ood': 4
    }
    
    def __init__(self, root_dir, subname='', version=None, download=True,
                 split_scheme="official", split_dict=None, split_names=None):
        self._split_names = split_names
        self._split_dict = split_dict 
        self._data_dir = root_dir 
        
        if split_dict is None:
            self.split_dict = RetrainDataset.DEFAULT_SPLITS 
        if split_names is None:
            self.split_names = RetrainDataset.DEFAULT_SPLIT_NAMES
        self.split_scheme = split_scheme
         
        # Load the dataset 
        ds = self.load_data(self.data_dir, subname)
        datadict, self._split_array = self.organize_data(ds, self.split_dict)

        self.features = list(datadict['features'])

         
        # Get the y values
        self._y_array = torch.LongTensor(datadict['labels'])

        y_unique = list(torch.unique(self._y_array).numpy().astype('str'))
        self._y_size = 1
        self._n_classes = len(y_unique) 
        self._collate = None
        self._metadata_fields = ['y']
        self._metadata_map = {'y': y_unique}
        self._metadata_array = self._y_array 
        if len(self._metadata_array.shape) == 1:
            self._metadata_array = self._metadata_array.unsqueeze(1)
         
        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['y'])
        )
        
    def __len__(self):
        return len(self.y_array)
        
    def __getitem__(self, idx): 
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        return self.get_input(idx), self.y_array[idx], self.metadata_array[idx]
        
    def get_input(self, idx):
        return self.text_array[idx]
    
    def load_data(self, data_dir, subname=''): 
        output = {'features':[], 'labels':[]}
        split_array = []
        for key in self.split_dict:
            filename = os.path.join(data_dir, key+subname+'.npz')
            if filename.is_file():
                load_data = np.load(filename)
                output['features'].append(load_data['features'])
                output['labels'].append(load_data['labels'])
                num = len(output['labels'][-1])
                split_array.append(self.split_dict[key]*np.ones(num,dtype=int)) 
         
        output['features'] = np.concatenate(output['features'], axis=0)
        output['labels'] = np.concatenate(output['labels'], axis=0)
        split_array = np.concatenate(split_array, axis=0)  
        return output, split_array

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        # return self.standard_eval(metric, y_pred, y_true)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
    
    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return CounterfactualTextSubset(self, split_idx, transform)

    @staticmethod
    def standard_eval(metric, y_pred, y_true):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = (
            f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        )
        return results, results_str
    
    @staticmethod
    def standard_group_eval(metric, grouper, y_pred, y_true, metadata, aggregate=True):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - grouper (CombinatorialGrouper): Grouper object that converts metadata into groups
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)
        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f'{metric.name}_{group_str}'] = group_metric
            results[f'count_{group_str}'] = group_counts
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            results_str += (
                f'  {grouper.group_str(group_idx)}  '
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {group_results[metric.group_metric_field(group_idx)]:5.3f}\n")
        results[f'{metric.worst_group_metric_field}'] = group_results[f'{metric.worst_group_metric_field}']
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'amazon', 'camelyon17'.
        """
        return self._dataset_name

    @property
    def data_dir(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_dir  

    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)

    @property
    def split_array(self):
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the (noisy) target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array
    
    @property
    def y_size(self):
        """
        The number of dimensions/elements in the target, i.e., len(y_array[i]).
        For standard classification/regression tasks, y_size = 1.
        For multi-task or structured prediction settings, y_size > 1.
        Used for logging and to configure models to produce appropriately-sized output.
        """
        return self._y_size

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def is_classification(self):
        """
        Boolean. True if the task is classification, and false otherwise.
        """
        return getattr(self, '_is_classification', (self.n_classes is not None))

    @property
    def is_detection(self):
        """
        Boolean. True if the task is detection, and false otherwise.
        """
        return getattr(self, '_is_detection', False)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

    @property
    def metadata_map(self):
        """
        An optional dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['hospital', 'y']
            metadata_map = {'hospital': ['East', 'West']}
        then if metadata_array[i, 0] == 0, the i-th data point belongs to the 'East' hospital
        while if metadata_array[i, 0] == 1, it belongs to the 'West' hospital.
        """
        return getattr(self, '_metadata_map', None)

class RetrainSubset(RetrainDataset):
    def __init__(self, dataset, indices, transform=None, do_transform_y=False):
        """
        This acts like `torch.utils.data.Subset`, but on `RetrainDataset`.
        We pass in `transform` (which is used for data augmentation) explicitly
        because it can potentially vary on the training vs. test subsets.
        `do_transform_y` (bool): When this is false (the default),
                                 `self.transform ` acts only on  `x`.
                                 Set this to true if `self.transform` should
                                 operate on `(x,y)` instead of just `x`.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_dict', '_split_names',
                           '_y_size', '_n_classes', 
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y, metadata

    def __len__(self):
        return len(self.indices)

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset._metadata_array[self.indices]

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)
