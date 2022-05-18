dataset_defaults = {
    'amazon': {
        'split_scheme': 'official',
        'model': 'distilbert-base-uncased',
        'transform': 'bert',
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 8,
        'unlabeled_batch_size': 8,
        'lr': 1e-5,
        'weight_decay': 0.01,
        'n_epochs': 3,
        'n_groups_per_batch': 2,
        'unlabeled_n_groups_per_batch': 2,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 1.0,
        'dann_penalty_weight': 1.0,
        'dann_featurizer_lr': 1e-6,
        'dann_classifier_lr': 1e-5,
        'dann_discriminator_lr': 1e-5,
        'loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'unlabeled_loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    'bdd100k': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'model_kwargs': {'pretrained': True},
        'loss_function': 'multitask_bce',
        'val_metric': 'acc_all',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'n_epochs': 10,
        'algo_log_metric': 'multitask_binary_accuracy',
        'transform': 'image_base',
        'process_outputs_function': 'binary_logits_to_pred',
    },
    'camelyon17': {
        'split_scheme': 'official', 
        'model': 'densenet121',
        'model_kwargs': {'pretrained': False},
        'transform': 'image_base',
        'target_resolution': (96, 96),
        'loss_function': 'cross_entropy',
        'groupby_fields': ['hospital'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'scheduler': None,
        'batch_size': 32,
        'unlabeled_batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.01,
        'n_epochs': 10,
        'n_groups_per_batch': 2,
        'unlabeled_n_groups_per_batch': 2,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'dann_penalty_weight': 0.1,
        'dann_featurizer_lr': 0.0001,
        'dann_classifier_lr': 0.001,
        'dann_discriminator_lr': 0.001,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    'camelyon17_noisy': {
        'split_scheme': 'official', 
        'noise_ratio': 0.2,
        'model': 'densenet121',
        'model_kwargs': {'pretrained': False},
        'transform': 'image_base',
        'target_resolution': (96, 96),
        'loss_function': 'cross_entropy',
        'groupby_fields': ['hospital'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'scheduler': None,
        'batch_size': 32,
        'unlabeled_batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.01,
        'n_epochs': 10,
        'n_groups_per_batch': 2,
        'unlabeled_n_groups_per_batch': 2,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'dann_penalty_weight': 0.1,
        'dann_featurizer_lr': 0.0001,
        'dann_classifier_lr': 0.001,
        'dann_discriminator_lr': 0.001,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    'celebA': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'model_kwargs': {'pretrained': True},
        'transform': 'image_base',
        'loss_function': 'cross_entropy',
        'groupby_fields': ['male', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'scheduler': None,
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 0.0,
        'n_epochs': 200,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'civilcomments': {
        'split_scheme': 'official',
        'model': 'distilbert-base-uncased',
        'transform': 'bert',
        'loss_function': 'cross_entropy',
        'groupby_fields': ['black', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'batch_size': 16,
        'unlabeled_batch_size': 16,
        'lr': 1e-5,
        'weight_decay': 0.01,
        'n_epochs': 5,
        'n_groups_per_batch': 1,
        'unlabeled_n_groups_per_batch': 1,
        'algo_log_metric': 'accuracy',
        'max_token_length': 300,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 10.0,
        'dann_penalty_weight': 1.0,
        'dann_featurizer_lr': 1e-6,
        'dann_classifier_lr': 1e-5,
        'dann_discriminator_lr': 1e-5,
        'loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'unlabeled_loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    "domainnet": {
        "split_scheme": "official",
        "dataset_kwargs": {
            "source_domain": "real",
            "target_domain": "sketch",
            "use_sentry": False,
        },
        "model": "resnet50",
        "model_kwargs": {"pretrained": True},
        "transform": "image_resize",
        "resize_scale": 256.0 / 224.0,
        "target_resolution": (224, 224),
        "loss_function": "cross_entropy",
        "groupby_fields": [
            "category",
        ],
        "val_metric": "acc_avg",
        "val_metric_decreasing": False,
        "batch_size": 96,
        "unlabeled_batch_size": 224,
        "optimizer": "SGD",
        "optimizer_kwargs": {
            "momentum": 0.9,
        },
        "lr": 0.0007035737028722148,
        "weight_decay": 1e-4,
        "n_epochs": 25,
        "n_groups_per_batch": 4,
        "unlabeled_n_groups_per_batch": 4,
        "irm_lambda": 1.0,
        "coral_penalty_weight": 1.0,
        "dann_penalty_weight": 1.0,
        "dann_featurizer_lr": 0.001,
        "dann_classifier_lr": 0.01,
        "dann_discriminator_lr": 0.01,
        "algo_log_metric": "accuracy",
        "process_outputs_function": "multiclass_logits_to_pred",
        "process_pseudolabels_function": "pseudolabel_multiclass_logits",
        "loader_kwargs": {
            "num_workers": 2,
            "pin_memory": True,
        },
    },
    'encode': {
        'split_scheme': 'official',
        'model': 'unet-seq',
        'model_kwargs': {'n_channels_in': 5},
        'loader_kwargs': {'num_workers': 1}, # pybigwig seems to have trouble with multiprocessing
        'transform': None,
        'loss_function': 'multitask_bce',
        'groupby_fields': ['celltype'],
        'val_metric': 'avgprec-macro_all',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'scheduler': 'MultiStepLR',
        'scheduler_kwargs': {'milestones':[3,6], 'gamma': 0.1},
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'n_epochs': 12,
        'n_groups_per_batch': 4,
        'algo_log_metric': 'multitask_binary_accuracy',
        'irm_lambda': 100.0,
        'coral_penalty_weight': 0.1,
    },
    'fmow': {
        'split_scheme': 'official',
        'dataset_kwargs': {
            'seed': 111,
            'use_ood_val': True
        },
        'model': 'densenet121',
        'model_kwargs': {'pretrained': True},
        'transform': 'image_base',
        'loss_function': 'cross_entropy',
        'groupby_fields': ['year',],
        'val_metric': 'acc_worst_region',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96},
        'batch_size': 32,
        'unlabeled_batch_size': 32,
        'lr': 0.0001,
        'weight_decay': 0.0,
        'n_epochs': 60,
        'n_groups_per_batch': 8,
        'unlabeled_n_groups_per_batch': 8,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'dann_penalty_weight': 1.0,
        'dann_featurizer_lr': 0.00001,
        'dann_classifier_lr': 0.0001,
        'dann_discriminator_lr': 0.0001,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    'iwildcam': {
        'loss_function': 'cross_entropy',
        'val_metric': 'F1-macro_all',
        'model_kwargs': {'pretrained': True},
        'transform': 'image_base',
        'target_resolution': (448, 448),
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'model': 'resnet50',
        'lr': 3e-5,
        'weight_decay': 0.0,
        'batch_size': 16,
        'unlabeled_batch_size': 16,
        'n_epochs': 12,
        'optimizer': 'Adam',
        'split_scheme': 'official',
        'scheduler': None,
        'groupby_fields': ['location',],
        'n_groups_per_batch': 2,
        'unlabeled_n_groups_per_batch': 2,
        'irm_lambda': 1.,
        'coral_penalty_weight': 10.,
        'dann_penalty_weight': 0.1,
        'dann_featurizer_lr': 3e-6,
        'dann_classifier_lr': 3e-5,
        'dann_discriminator_lr': 3e-5,
        'no_group_logging': True,
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    'iwildcam_noisy': {
        'noise_ratio': 0.2,
        'loss_function': 'cross_entropy',
        'val_metric': 'F1-macro_all',
        'model_kwargs': {'pretrained': True},
        'transform': 'image_base',
        'target_resolution': (448, 448),
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'model': 'resnet50',
        'lr': 3e-5,
        'weight_decay': 0.0,
        'batch_size': 16,
        'unlabeled_batch_size': 16,
        'n_epochs': 12,
        'optimizer': 'Adam',
        'split_scheme': 'official',
        'scheduler': None,
        'groupby_fields': ['location',],
        'n_groups_per_batch': 2,
        'unlabeled_n_groups_per_batch': 2,
        'irm_lambda': 1.,
        'coral_penalty_weight': 10.,
        'dann_penalty_weight': 0.1,
        'dann_featurizer_lr': 3e-6,
        'dann_classifier_lr': 3e-5,
        'dann_discriminator_lr': 3e-5,
        'no_group_logging': True,
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },
    'ogb-molpcba': {
        'split_scheme': 'official',
        'model': 'gin-virtual',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'loss_function': 'multitask_bce',
        'groupby_fields': ['scaffold',],
        'val_metric': 'ap',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'unlabeled_batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'unlabeled_n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'dann_penalty_weight': 0.1,
        'dann_featurizer_lr': 1e-3,
        'dann_classifier_lr': 1e-2,
        'dann_discriminator_lr': 1e-2,
        'noisystudent_add_dropout': False,
        'no_group_logging': True,
        'algo_log_metric': 'multitask_binary_accuracy',
        'process_outputs_function': None,
        'process_pseudolabels_function': 'pseudolabel_binary_logits',
        'loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
    },
    'py150': {
        'split_scheme': 'official',
        'model': 'code-gpt-py',
        'loss_function': 'lm_cross_entropy',
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'AdamW',
        'optimizer_kwargs': {'eps':1e-8},
        'lr': 8e-5,
        'weight_decay': 0.,
        'n_epochs': 3,
        'batch_size': 6,
        'groupby_fields': ['repo',],
        'n_groups_per_batch': 2,
        'irm_lambda': 1.,
        'coral_penalty_weight': 1.,
        'no_group_logging': True,
        'algo_log_metric': 'multitask_accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'poverty': {
        'split_scheme': 'official',
        'dataset_kwargs': {
            'no_nl': False,
            'fold': 'A',
            'use_ood_val': True
        },
        'model': 'resnet18_ms',
        'model_kwargs': {'num_channels': 8},
        'transform': 'poverty',
        'loss_function': 'mse',
        'groupby_fields': ['country',],
        'val_metric': 'r_wg',
        'val_metric_decreasing': False,
        'algo_log_metric': 'mse',
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96},
        'batch_size': 64,
        'unlabeled_batch_size': 64,
        'lr': 0.001,
        'weight_decay': 0.0,
        'n_epochs': 200,
        'n_groups_per_batch': 8,
        'unlabeled_n_groups_per_batch': 4,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'dann_penalty_weight': 0.1,
        'dann_featurizer_lr': 0.0001,
        'dann_classifier_lr': 0.001,
        'dann_discriminator_lr': 0.001,
        'process_outputs_function': None,
        'process_pseudolabels_function': 'pseudolabel_identity',
    },
    'waterbirds': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'transform': 'image_resize_and_center_crop',
        'resize_scale': 256.0/224.0,
        'model_kwargs': {'pretrained': True},
        'loss_function': 'cross_entropy',
        'groupby_fields': ['background', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum':0.9},
        'scheduler': None,
        'batch_size': 128,
        'lr': 1e-5,
        'weight_decay': 1.0,
        'n_epochs': 300,
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'yelp': {
        'split_scheme': 'official',
        'model': 'bert-base-uncased',
        'transform': 'bert',
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 8,
        'lr': 2e-6,
        'weight_decay': 0.01,
        'n_epochs': 3,
        'n_groups_per_batch': 2,
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'imdb': {
        'split_scheme': 'official',
        'model': 'bert-base-uncased',
        'transform': 'bert',
        'groupby_fields': ['y'],
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 16,
        'lr': 1e-5,
        'weight_decay': 0,
        'n_epochs': 45,
        'n_groups_per_batch': 2,
        'process_outputs_function': 'multiclass_logits_to_pred',
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False, 
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': 'cosine_schedule_with_warmup',
        'scheduler_kwargs': {'num_warmup_steps': 5415}, 
    },
    'imdb_sents': {
        'split_scheme': 'official',
        'model': 'bert-base-uncased',
        'transform': 'bert', 
        'groupby_fields': ['y'],
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 16,
        'lr': 1e-5,
        'weight_decay': 0,
        'n_epochs': 30,
        'n_groups_per_batch': 2,
        'process_outputs_function': 'multiclass_logits_to_pred',
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False, 
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': 'cosine_schedule_with_warmup',
        'scheduler_kwargs': {'num_warmup_steps': 5415}, 
    },
    'kindle': {
        'split_scheme': 'official',
        'model': 'bert-base-uncased',
        'transform': 'bert',
        'groupby_fields': ['y'],
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 16,
        'lr': 1e-5,
        'weight_decay': 0,
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'process_outputs_function': 'multiclass_logits_to_pred',
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False, 
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': 'cosine_schedule_with_warmup',
        'scheduler_kwargs': {'num_warmup_steps': 5415}, 
    },
    'sqf': {
        'split_scheme': 'all_race',
        'model': 'logistic_regression',
        'transform': None,
        'model_kwargs': {'in_features': 104},
        'loss_function': 'cross_entropy',
        'groupby_fields': ['y'],
        'val_metric': 'precision_at_global_recall_all',
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': None,
        'batch_size': 4,
        'lr': 5e-5,
        'weight_decay': 0,
        'n_epochs': 4,
        'process_outputs_function': None,
    },
    'rxrx1': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'model_kwargs': {'pretrained': True},
        'transform': 'rxrx1',
        'target_resolution': (256, 256),
        'loss_function': 'cross_entropy',
        'groupby_fields': ['experiment'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': 'cosine_schedule_with_warmup',
        'scheduler_kwargs': {'num_warmup_steps': 5415},
        'batch_size': 72,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'n_groups_per_batch': 9,
        'coral_penalty_weight': 0.1,
        'irm_lambda': 1.0,
        'n_epochs': 90,
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'rxrx1_noisy': {
        'split_scheme': 'official',
        'noise_ratio' : 0.2,
        'model': 'resnet50',
        'model_kwargs': {'pretrained': True},
        'transform': 'rxrx1',
        'target_resolution': (256, 256),
        'loss_function': 'cross_entropy',
        'groupby_fields': ['experiment'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': 'cosine_schedule_with_warmup',
        'scheduler_kwargs': {'num_warmup_steps': 5415},
        'batch_size': 72,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'n_groups_per_batch': 9,
        'coral_penalty_weight': 0.1,
        'irm_lambda': 1.0,
        'n_epochs': 90,
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'globalwheat': {
        'split_scheme': 'official',
        'model': 'fasterrcnn',
        'transform': 'image_base',
        'model_kwargs': {
            'n_classes': 1,
            'pretrained': True
        },
        'loss_function': 'fasterrcnn_criterion',
        'groupby_fields': ['session'],
        'val_metric': 'detection_acc_avg_dom',
        'val_metric_decreasing': False,
        'algo_log_metric': None, # TODO
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': None,
        'batch_size': 4,
        'unlabeled_batch_size': 4,
        'lr': 1e-5,
        'weight_decay': 1e-3,
        'n_epochs': 12,
        'noisystudent_add_dropout': False,
        'self_training_threshold': 0.5,
        'loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'process_outputs_function': None,
        'process_pseudolabels_function': 'pseudolabel_detection_discard_empty',
    }
}

##########################################
### Split-specific defaults for Amazon ###
##########################################

amazon_split_defaults = {
    'official':{
        'groupby_fields': ['user'],
        'val_metric': '10th_percentile_acc',
        'val_metric_decreasing': False,
        'no_group_logging': True,
    },
    'user':{
        'groupby_fields': ['user'],
        'val_metric': '10th_percentile_acc',
        'val_metric_decreasing': False,
        'no_group_logging': True,
    },
    'time':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
    'time_baseline':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
}

user_baseline_splits = [
    'A1CNQTCRQ35IMM_baseline', 'A1NE43T0OM6NNX_baseline', 'A1UH21GLZTYYR5_baseline', 'A20EEWWSFMZ1PN_baseline',
    'A219Y76LD1VP4N_baseline', 'A37BRR2L8PX3R2_baseline', 'A3JVZY05VLMYEM_baseline', 'A9Q28YTLYREO7_baseline',
    'ASVY5XSYJ1XOE_baseline', 'AV6QDP8Q0ONK4_baseline'
    ]
for split in user_baseline_splits:
    amazon_split_defaults[split] = {
        'groupby_fields': ['user'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        }

category_splits = [
    'arts_crafts_and_sewing_generalization', 'automotive_generalization',
    'books,movies_and_tv,home_and_kitchen,electronics_generalization', 'books_generalization', 'category_subpopulation',
    'cds_and_vinyl_generalization', 'cell_phones_and_accessories_generalization', 'clothing_shoes_and_jewelry_generalization',
    'digital_music_generalization', 'electronics_generalization', 'grocery_and_gourmet_food_generalization',
    'home_and_kitchen_generalization', 'industrial_and_scientific_generalization', 'kindle_store_generalization',
    'luxury_beauty_generalization', 'movies_and_tv,books,home_and_kitchen_generalization', 'movies_and_tv,books_generalization',
    'movies_and_tv_generalization', 'musical_instruments_generalization', 'office_products_generalization',
    'patio_lawn_and_garden_generalization', 'pet_supplies_generalization', 'prime_pantry_generalization',
    'sports_and_outdoors_generalization', 'tools_and_home_improvement_generalization', 'toys_and_games_generalization',
    'video_games_generalization',
    ]
for split in category_splits:
    amazon_split_defaults[split] = {
        'groupby_fields': ['category'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        }

########################################
### Split-specific defaults for Yelp ###
########################################

yelp_split_defaults = {
    'official':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
    'user':{
        'groupby_fields': ['user'],
        'val_metric': '10th_percentile_acc',
        'val_metric_decreasing': False,
        'no_group_logging': True,
    },
    'time':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
    'time_baseline':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
}

###############################
### Split-specific defaults ###
###############################

split_defaults = {
    'amazon': amazon_split_defaults,
    'yelp': yelp_split_defaults,
}
