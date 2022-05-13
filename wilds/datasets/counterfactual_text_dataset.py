import os 
import io, time 
from IPython.display import display

import numpy as np
import pandas as pd
import torch
import pickle, random, re
from collections import Counter, defaultdict
from PyDictionary import PyDictionary
dictionary=PyDictionary()

from wilds.common.metrics.all_metrics import Accuracy



class Counterfactual:
    def __init__(self, df_train, df_test, moniker):
        display(df_train.head(1))
        self.moniker = moniker
        self.train = df_train
        self.test = df_test
        
def load_data(data_dir, moniker): 
    if(moniker == 'kindle'):
        load_path = os.path.join(data_dir, "ds_kindle.pkl") 
    elif(moniker == 'imdb'):
        load_path = os.path.join(data_dir, "ds_imdb_para.pkl") 
    elif(moniker == 'imdb_sents'):
        load_path = os.path.join(data_dir, "ds_imdb_sent.pkl")  
    ds = pickle.load(open(load_path, "rb"))
    return ds

def organize_data(ds, create_id_val=False):
    """
    Organize data for easy use in the evaluation
    train, valid, test, test_ood
    """
    ds.train['label'] = (ds.train.label.values == 1).astype(int)
    
    output = defaultdict(dict)
    if(ds.moniker == 'imdb'):
        # Counterfactual by human
        output['val'] = {'text':ds.train.ct_text_amt.values, 'label':(ds.train.ct_label.values == 1).astype(int)}
    elif(ds.moniker == 'imdb_sents'):
        # Counterfactual by human
        output['val'] = {'text':ds.train_ct.text.values, 'label':(ds.train_ct.label.values == 1).astype(int)}
    elif(ds.moniker == 'kindle'):
        # Auto-generated counterfactual training samples
        # Causal terms annotated from whole vocabulary
        flag = 'all_causal'
        col = 'all_causal'
        ds.train['len_ct_text_'+col] = ds.train['ct_text_'+col].apply(lambda x: len(x.strip()))
        df_train_ct_text_flag = ds.train[ds.train['len_ct_text_'+col] > 0]
        output['val'] = {'text':df_train_ct_text_flag['ct_text_'+col].values, 'label':df_train_ct_text_flag.ct_label.values}
    
    if create_id_val: 
        if ds.moniker == "imdb": 
            n_classes = 2
            # number of samples for validation data
            num_val = 500
        elif ds.moniker == "imdb_sents": 
            n_classes = 2
            # number of samples for validation data
            num_val = 1000
        elif ds.moniker == 'kindle':
            n_classes = 2
            # number of samples for validation data
            num_val = 1000 
        
        uniqueLabel = np.unique(ds.train.label.values)    
        smpID = []
        for label in uniqueLabel:
            AllID = ds.train.index[ds.train.label.values == label].tolist()
            smpID += list(np.random.choice(AllID, int(num_val/n_classes), replace=False))

        valID = ds.train.index.isin(smpID)
        val_data = ds.train.iloc[valID]
        output['id_val'] = {'text':val_data.text.values, 'label':val_data.label.values}
        train_data = ds.train.iloc[~valID]
    else:
        train_data = ds.train
    
    output['train'] = {'text':train_data.text.values, 'label':train_data.label.values}
    
        
    # Test and Test OOD
    # Contert it to binary labels  
    label = (ds.test.label.values == 1).astype(int)  
    if ds.moniker == 'imdb':
        # Convert it to binary labels 
        label_cft = (ds.test.ct_label.values == 1).astype(int)  
        # Original
        output['id_test'] = {'text': ds.test.text.values, 'label': label}
        # Counterfactual 
        output['test'] = {'text': ds.test.ct_text_amt.values, 'label': label_cft}
    if ds.moniker == 'imdb_sents':
        # Convert it to binary labels  
        label_cft = (ds.test_ct.label.values == 1).astype(int)
        # Original
        output['id_test'] = {'text': ds.test.text.values, 'label': label} 
        # Counterfactual 
        output['test'] = {'text': ds.test_ct.text.values, 'label': label_cft}
    elif ds.moniker == 'kindle': 
        # Contert it to binary labels  
        label_cft = (ds.test.ct_label.values == 1).astype(int)
        # Original
        output['id_test'] = {'text': ds.test.text.values, 'label': label} 
        # Counterfactual
        output['test'] = {'text': ds.test.ct_text_amt.values, 'label': label_cft}
 
    return output

class CtfTextDataset:   
    DEFAULT_SPLIT_NAMES = {
        'train': 'Train',
        'id_val': 'Validation (ID)',
        'id_test': 'Test',
        'val' : 'Validation (OOD)'
        'test': 'Test (OOD)',
    }
    def __init__(self, datadict, transform):
  
        self.transform = transform 
        # Get the y values
        self.y_array = torch.LongTensor(datadict['label'])
        self.text_array = list(datadict['text'])
        self.y_size = 1
        self.n_classes = 2
        self.is_classification = True
        self.collate = None
        
    def __len__(self):
        return len(self.y_array)
        
    def __getitem__(self, idx): 
        return self.transform(self.get_input(idx)), self.y_array[idx]
        
    def get_input(self, idx):
        return self.text_array[idx]
    
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
    
    def eval(self, y_pred, y_true, metadata=None, prediction_fn=None):
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
        return self.standard_eval(metric, y_pred, y_true)
    
    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        e.g., {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.
        Keys should match up with split_dict.
        """
        return getattr(self, '_split_names', CtfTextDataset.DEFAULT_SPLIT_NAMES)
