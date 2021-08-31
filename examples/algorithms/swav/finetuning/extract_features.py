import argparse
import os
import pdb
import pickle
import sys

import torch
import numpy as np
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from examples.algorithms.swav.src.model import SwAVModel
from examples.algorithms.swav.src.utils import ParseKwargs, populate_defaults_for_swav
from examples.models.initializer import initialize_model
from examples.transforms import initialize_transform


def get_model(config):
    d_out = 1  # this can be arbitrary; final layer is discarded for SwAVModel
    base_model = initialize_model(config, d_out, **config.model_kwargs)
    model = SwAVModel(base_model, output_dim=0, eval_mode=True)
    checkpoint = torch.load(
        os.path.join(config.run_dir, "checkpoints", f"ckp-{config.ckpt_epoch}.pth"),
        map_location="cpu",
    )
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    pdb.set_trace()
    assert len(missing_keys) == 0, f'Was unable to match keys: {",".join(missing_keys)}'
    return model.eval().cuda()


def get_data_loaders(config):
    dataset = wilds.get_dataset(
        dataset=config.dataset,
        root_dir=config.root_dir,
        download=True,
        **config.dataset_kwargs,
    )
    train_transform = initialize_transform(
        transform_name=config.train_transform, config=config, dataset=dataset
    )
    train_data = dataset.get_subset("train", transform=train_transform)
    eval_transform = initialize_transform(
        transform_name=config.eval_transform,
        config=config,
        dataset=dataset,
    )
    test_data = dataset.get_subset(config.eval_split, transform=eval_transform)
    train_loader = get_train_loader(
        "standard", train_data, batch_size=config.batch_size, **config.loader_kwargs
    )
    test_loader = get_eval_loader(
        "standard", test_data, batch_size=config.batch_size, **config.loader_kwargs
    )
    return train_loader, test_loader


def get_features(model, data_loaders):
    feats = []
    for loader in data_loaders:
        features_list, labels_list = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.cuda()
                features = model(x)
                features_list.append(features.detach().cpu().numpy())
                labels_list.append(y.detach().numpy())
        features = np.squeeze(np.concatenate(features_list))
        labels = np.concatenate(labels_list)
        feats.append([features, labels])
    return feats


def main():
    config = populate_defaults_for_swav(args)
    model = get_model(config)
    data_loaders = get_data_loaders(config)
    features = get_features(model, data_loaders)

    # Save extracted features to log_dir
    os.makedirs(config.log_dir, exist_ok=True)
    output_file_path = os.path.join(
        config.log_dir, f"features_and_labels_{config.ckpt_epoch}.pickle"
    )
    pickle.dump(features, open(output_file_path, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # SwAV checkpoint args
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="The SwAV run directory.",
    )
    parser.add_argument(
        "--ckpt_epoch",
        type=int,
        default=399,
        help="The epoch of the checkpoint in the checkpoints/ folder.",
    )
    # Dataset args
    parser.add_argument(
        "-d", "--dataset", required=True, choices=wilds.unlabeled_datasets
    )
    parser.add_argument("--dataset_kwargs", nargs="*", action=ParseKwargs, default={})
    parser.add_argument("--loader_kwargs", nargs="*", action=ParseKwargs, default={})
    parser.add_argument(
        "--root_dir",
        required=True,
        help="The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--eval_split",
        default="val",
        help="The split of the WILDS dataset to use for evaluation.",
    )
    # Model args
    parser.add_argument(
        "--model",
        type=str,
        help="Convnet architecture.",
    )
    parser.add_argument(
        "--model_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="Keyword arguments for model initialization passed as key1=value1 key2=value2",
    )
    # Where to save extracted features
    parser.add_argument(
        "--log_dir",
        type=str,
        default=".",
        help="The directory where to save the extracted features to.",
    )

    # Parse args and run this script
    args = parser.parse_args()
    main()
