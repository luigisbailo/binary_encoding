import sys
import yaml
import argparse
import pickle
import importlib
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize

import binary_encoding.networks as networks
from binary_encoding.trainer import Trainer


def parse_config(config_file):
    """
    Parse the given config file and return the data.

    Args:
        config_file (str): The path to the config file.

    Returns:
        dict: The parsed data from the config file.

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there is an error parsing the config file.
    """
    try:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def convert_bool(dictionary):
    """
    Converts string values 'true' and 'false' in the dictionary to boolean
    values.

    Args:
        dictionary (dict): The dictionary to be converted.

    Returns:
        dict: The dictionary with string values converted to boolean values.
    """
    for key, value in dictionary.items():
        if type(value) is str:
            if value.lower() == 'true':
                dictionary[key] = True
            elif value.lower() == 'false':
                dictionary[key] = False
    return dictionary


def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset.

    Args:
        dataset: The dataset to compute the mean and standard deviation for.

    Returns:
        mean: The mean of the dataset.
        std: The standard deviation of the dataset.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(loader))[0].numpy()
    mean = np.mean(data, axis=(0, 2, 3))
    std = np.std(data, axis=(0, 2, 3))
    return mean, std


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("CUDA is not available. Training on CPU.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', required=False)
    parser.add_argument('--lr', required=False)
    parser.add_argument('--dropout', required=False)
    parser.add_argument('--augment', required=False)
    parser.add_argument('--encoding-metrics', required=False)
    parser.add_argument('--store-penultimate', required=False)

    args = parser.parse_args()

    config_file = args.config
    results_dir = args.results_dir
    dataset_dir = args.dataset_dir
    model = args.model
    sample = args.sample
    lr = args.lr
    dropout = args.dropout
    augment = args.augment
    encoding_metrics = args.encoding_metrics
    store_penultimate = args.store_penultimate

    configs = parse_config(config_file)
    architecture = configs['architecture']
    training_hypers = configs['training']['hypers']
    name_dataset = configs['dataset']['name']

    # Overwrite learning rate if provided as argument
    if lr:
        if lr.startswith('.'):
            lr = '0'+lr
        training_hypers['lr'] = float(lr)

    # If encoding_metrics is not provided, default to False
    encoding_metrics = encoding_metrics.lower() if encoding_metrics else False
    if encoding_metrics not in ['true', 'false']:
        print("Invalid value for encoding_metrics. Defaulting to False.")
        encoding_metrics = False
    else:
        encoding_metrics = encoding_metrics == 'true'
        
    # If dropout is not provided, default to False
    dropout = dropout.lower() if dropout else False
    if dropout not in ['true', 'false']:
        print("Invalid value for dropout. Defaulting to False.")
        dropout = False
    else:
        dropout = dropout == 'true'
    architecture['hypers']['dropout_backbone'] = dropout

    # If augment is not provided, default to False
    augment = augment.lower() if augment else False
    if augment not in ['true', 'false']:
        print("Invalid value for augment. Defaulting to False.")
        augment = False
    else:
        augment = augment == 'true'
    
    # If store_penultimate is not provided, default to False
    store_penultimate = store_penultimate.lower() if store_penultimate else False
    if store_penultimate not in ['true', 'false']:
        print("Invalid value for store_penultimate. Defaulting to False.")
        store_penultimate = False
    else:
        store_penultimate = store_penultimate == 'true'

    transform = transforms.Compose([transforms.ToTensor()])
    torch_module = importlib.import_module("torchvision.datasets")

    if (name_dataset == 'SVHN'):
        torch_dataset = getattr(torch_module, name_dataset)
        trainset = torch_dataset(
            str(dataset_dir), split='train', download=True, transform=transform)
    else :
        torch_dataset = getattr(torch_module, name_dataset)
        trainset = torch_dataset(
            str(dataset_dir), train=True, download=True, transform=transform)

    
    if (name_dataset == 'SVHN'):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(trainset[0][0][0][0].shape[0], padding=4),
            Resize((32, 32)),  

        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            Resize((32, 32)),  

        ])
    else:
        trainset_mean, trainset_std = compute_mean_std(trainset)
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(trainset[0][0][0][0].shape[0], padding=4),
            
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(trainset_mean, trainset_std),
        ])

    
    if (name_dataset == 'SVHN'):
        trainset = torch_dataset(
        str(dataset_dir), split='train', download=True, transform=transform_train)
        testset = torch_dataset(
            str(dataset_dir), split='test', download=True, transform=transform_test)
    else:
        trainset = torch_dataset(
            str(dataset_dir), train=True, download=True, transform=transform_train)
        testset = torch_dataset(
            str(dataset_dir), train=False, download=True, transform=transform_test)

    input_dims = trainset[0][0].numel()
    if (name_dataset == 'SVHN'):
        num_classes = 10
    else:
        num_classes = len(set(trainset.classes))
    
    training_hypers = convert_bool(training_hypers)
    architecture['hypers'] = convert_bool(architecture['hypers'])

    print('Training ' + str(model) + ' architecture:')
    print('Learning rate: ', training_hypers['lr'])

    classifier = getattr(networks, architecture['backbone'])(
        model=model,
        architecture=architecture,
        num_classes=num_classes,
        input_dims=input_dims,
        )
    if torch.cuda.device_count() > 1:
        print('data parallel')
        classifier = torch.nn.DataParallel(classifier)
    if torch.cuda.is_available():
        classifier = classifier.to(device)


    results = Trainer(
        device=device,
        network=classifier,
        trainset=trainset,
        testset=testset,
        training_hypers=training_hypers,
        model=model,
        encoding_metrics=encoding_metrics,
        store_penultimate=store_penultimate,
        verbose=True
        ).fit()
    
    print('finished_training')
    
    results['training_hypers'] = training_hypers
    results['architecture'] = architecture
    
    if dropout:
        if sample:
            file_name = '/dropout_' + model + '_' + sample + '.pkl'
        else:
            file_name = '/dropout_' + model + '.pkl'
    else:
        if sample:
            file_name = '/' + model + '_' + sample + '.pkl'
        else:
            file_name = '/' + model + '_' + '.pkl'
    
    print(file_name)
            
    with open(str(results_dir) + file_name, 'wb') as file:
        pickle.dump(results, file)
