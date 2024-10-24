import os
import random
import sys
import tempfile
import time
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils
from transformers import AutoModel, AutoTokenizer
import yaml

from data.dataset import DrugDataset
from models.model import MLPModel, CNNModel

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(stream = sys.stdout, level=logging.INFO)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
        
def run(config):
    # Set seed
    seed = config['task']['seed']
    logging.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_name= config['dataset']['use_name']
    use_description = config['dataset']['use_description']
    use_smile = config['dataset']['use_smile']
    use_formula = config['dataset']['use_formula']

    # Init model
    if config['model']['model_type'] == "MLPModel":
        model = MLPModel(use_name=use_name, use_description=use_description, use_smile=use_smile, use_formula=use_formula)
    elif config['model']['model_type'] == "CNNModel":
        model = CNNModel(use_name=use_name, use_description=use_description, use_smile=use_smile, use_formula=use_formula)
    else:
        raise ValueError(f"Model type {config['model']['model_type']} not supported")
    
    logging.info(f"Model: {model}")
    world_size = torch.cuda.device_count()
#     model, lr, num_epochs, save_model_path, dataloaders, dataset_sizes,  world_size
    mp.spawn(train_model,
             args=(model, world_size, config),
             nprocs=world_size,
             join=True)

    
def train_model(rank, model, world_size, config):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Unpack config
    lr = config['optimizer']['lr']
    num_epochs = config['task']['epochs']
    load_model_path = config['task']['load_checkpoint_path']
    save_model_path = config['task']['save_checkpoint_path']
    batch_size = config['task']['batch_size']
    optimizer_type = config['optimizer']['optimizer_type']
    do_train = config['task']['train']
    do_test = config['task']['test']

    # Handle dataloader
    train_dataset = config['train_dataset']
    test_dataset = config['test_dataset']
    dataset_sizes = {
            "train": len(train_dataset),
            "test": len(test_dataset),
            "dev": len(test_dataset)
        }
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler=DistributedSampler(train_dataset))
    val_loader = DataLoader(test_dataset, batch_size = batch_size, sampler=DistributedSampler(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, sampler=DistributedSampler(test_dataset))
    dataloaders = {'train':train_loader, 'val':val_loader, 'test': test_loader}
    
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer type {optimizer_type} not supported")
    
    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            # Use DistributedSampler for distributed data loading
            if phase == 'train':
                dataloaders[phase].sampler.set_epoch(epoch)

            total_batches = len(dataloaders[phase])
            # Iterate over data
            for data in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Progress", total=total_batches):
                # TODO: Re-check it and implement it in the model!
                # embed1 = embed1.to(rank).to(torch.float32)
                # embed2 = embed2.to(rank).to(torch.float32)
                labels = data[-1].to(rank).long()

                optimizer.zero_grad()
                if phase == 'train':
                    outputs = model(data)
                else:
                    with torch.no_grad():
                        outputs = model(data)

                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                loss = criterion(outputs, labels)

                # Backward pass and optimization only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if rank == 0:  # Only log from the main process
                logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                wandb.log({
                    f'{phase}/loss': epoch_loss,
                    f'{phase}/acc': epoch_acc
                })

                if phase == 'val':
                    report = classification_report(
                        y_true=all_labels, 
                        y_pred=all_preds, 
                        labels=[0, 1, 2, 3],  
                        target_names=['No Interaction', 'New Adverse', 'Antagonism', 'Synergism'],  
                        digits=4,  
                        output_dict=True
                    )
                    for label, metrics in report.items():
                        if isinstance(metrics, dict):
                            for metric_name, metric_value in metrics.items():
                                wandb.log({f"{label}/{metric_name}": metric_value})
                        else:
                            wandb.log({label: metrics})
                    logging.info(report)
                    logging.info("Without the negative label:")
                    report = classification_report(
                        y_true=all_labels, 
                        y_pred=all_preds, 
                        labels=[1, 2, 3],  
                        target_names=['New Adverse', 'Antagonism', 'Synergism'],  
                        digits=4,  
                        output_dict=True
                    )
                    logging.info(report)

        if rank == 0 and save_model_path:
            torch.save(model.state_dict(), config['output_dir'] + save_model_path)
            
    cleanup()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml config file')
    args = parser.parse_args()
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # FILE
    train_dataset = torch.load(config['train_dataset_path'])
    test_dataset = torch.load(config['test_dataset_path'])
    id2embedding = torch.load(config['id2embedding_path'])
    train_dataset.id2embedding = id2embedding
    test_dataset.id2embedding = id2embedding

    config['train_dataset'] = train_dataset
    config['test_dataset'] = test_dataset

    # Init wandb
    if config['wandb']['log']:
        wandb.login(key=config['wandb']['api_key'])
        user = config['wandb']['user']
        project = config['wandb']['project_name']
        display_name = config['wandb']['display_name']
        wandb.init(entity=user, project=project, name=display_name)

    n_gpus = torch.cuda.device_count()
    logging.info(f"total GPUs: {n_gpus}")
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    # TODO: Check if it can run with 1 gpu
    world_size = n_gpus
    run(config)
    
    
# def launch(seed, run_name, train_dataset, test_dataset, model="MLP", lr=1e-4, num_epochs=1, save_model_path="checkpoint.pt"):
#     # TODO: GET IT FROM CONFIG
#     wandb.login(key=secret_value_0)
#     user = "re-2023"
#     project = "DDI_Oct_2024"
#     display_name = run_name

#     wandb.init(entity=user, project=project, name=display_name)
#     logging.info("Seed: ", seed)
    
#     train_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True)
#     val_loader = DataLoader(test_dataset, batch_size = 128, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size = 128, shuffle=False)
    
#     dataloader = {'train':train_loader, 'val':val_loader, 'test': test_loader}
#     logging.info("Model", model)
#     
    
#     if model=="MLP":
#         model = MLPModel()
#     else:
#         model = CNNModel()
#     logging.info(model)
#     dataset_sizes = {
#         "train": len(train_dataset),
#         "test": len(test_dataset),
#         "dev": len(test_dataset)
#     }

    
#     wandb.finish()