import os
import random
import sys
import tempfile
import time

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils
from transformers import AutoModel, AutoTokenizer

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

class DrugPairDataset():
    def __init__(self, data_path, 
                 drug_description_dict,
                 train_ratio=0.8,
                 type_info="description", 
                 handle_direct=False, 
                 column_names=['drug1', 'drug2', 'label', 'origin_label'], 
                 seed=42,
                 undirect=False,
                false_set_limit=10000):
        self.data_path = data_path
        self.drug_description_dict = drug_description_dict
        self.train_ratio= train_ratio
        self.handle_direct = handle_direct
        self.type_info=type_info
        self.seed= seed
        self.false_set_limit = false_set_limit
        self.data = []
        self.label = []
        self.label_set = set()
        self.undirect = undirect

        self.handle_dataset(data_path, handle_direct, column_names)
        # Split train test
        
    def handle_dataset(self, path, handle_direct, column_names):
        """
        Receive a csv file as input, including columns: Drug1, Drug2, label, origin_label.
        """
        drug1, drug2, label, origin_label = column_names
        df = pd.read_csv(path)
        self.drugs_set = set()
        self.pairs_set = set()
        self.undirect_pairs_set = set()

        # Add to set and data
        for i in tqdm(range(len(df[drug1]))):
            drug1_id = df[drug1][i].split("::")[-1]
            drug2_id = df[drug2][i].split("::")[-1]

            drug1_info = get_drug_description(self.drug_description_dict, drug1_id)
            drug2_info = get_drug_description(self.drug_description_dict, drug2_id)
            
            int_label = df[label][i]
            # Check before add:
            if self.undirect and ((drug1_info, drug2_info) in self.pairs_set or (drug2_info, drug1_info) in self.pairs_set):
                continue
            self.data.append([drug1_info, drug2_info])
            self.pairs_set.add((drug1_info, drug2_info))
            
            if int_label == 'New Adverse':
                self.undirect_pairs_set.add((drug1_info, drug2_info))
            elif self.undirect:
                self.undirect_pairs_set.add((drug1_info, drug2_info))

            
            self.drugs_set.add(drug1_info)
            self.drugs_set.add(drug2_info)
            self.label.append(int_label)
            self.label_set.add(int_label)
            
        self.label_set = {item: index+1 for index, item in enumerate(self.label_set)}
        print("The label set is:\n", self.label_set)
        print("Data len from csv:", )
        self.label = [self.label_set[i] for i in self.label]
        
        # Add false data
        false_data = self.get_false_data()
        false_label = [0] * len(false_data)
        self.data.extend(false_data)
        self.label.extend(false_label)
        self.label_set['No Interaction'] = 0
        
        # Split train test
        self.shuffle_list = list(range(len(self.label)))
        random.shuffle(self.shuffle_list)
        self.shuffle_train_list = self.shuffle_list[:int(len(self.label)*self.train_ratio)]
        self.shuffle_test_list = self.shuffle_list[int(len(self.label)*self.train_ratio):]
        self.train_dataset = []
        self.train_label = []
        self.test_dataset = []
        self.test_label = []
        
        for i in self.shuffle_train_list:
            self.train_dataset.append(self.data[i])
            self.train_label.append(self.label[i])
        
        for i in self.shuffle_test_list:
            self.test_dataset.append(self.data[i])
            self.test_label.append(self.label[i])
        
    def get_false_data(self):
        false_data = []
        amount = 0
        drugs_amount = len(self.drugs_set)
        drugs_list = list(self.drugs_set)
        random.seed(self.seed)
        with tqdm(total=self.false_set_limit) as pbar:
            while True:
                idx1, idx2 = random.randint(0, drugs_amount-1), random.randint(0, drugs_amount-1)
                if (drugs_list[idx1], drugs_list[idx2]) in self.pairs_set:
                    continue
                if (drugs_list[idx2], drugs_list[idx1]) in self.undirect_pairs_set: # Correct logic, but tricky!
                    # nếu cặp reverse đã tồn tại và là nhãn vô hướng thì không cho là negative sample
                    continue
                false_data.append([drugs_list[idx1], drugs_list[idx2]])
                self.pairs_set.add((drugs_list[idx1], drugs_list[idx2]))
                amount+=1
                pbar.update(1)
                if amount>=self.false_set_limit:
                    break
        return false_data

def create_drug_description_dict(csv_file):
    """Creates a dictionary mapping drug IDs to their descriptions."""
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    # Replace null descriptions with drug names
    df['description'] = df['description'].fillna(df['name'])
    # Create a dictionary from the DataFrame
    drug_description_dict = df.set_index('drug-id')['description'].to_dict()
    return drug_description_dict

class DrugDataset(Dataset):
    def __init__(self, data, labels, id_to_cls_token_embed):
        self.data = data
        self.labels = labels
        self.id_to_cls_token_embed = id_to_cls_token_embed
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract the label and the pair (name1, name2) from the DataFrame
        
        embed1 = self.id_to_cls_token_embed[self.data[idx][0]]
        embed2 = self.id_to_cls_token_embed[self.data[idx][1]]
        label = self.labels[idx]
        
        return embed1, embed2, label
    
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '5554'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()


def launch(seed, run_name, train_dataset, test_dataset, model="MLP", lr=1e-4, num_epochs=1, save_model_path="checkpoint.pt"):
    # TODO: GET IT FROM CONFIG
    wandb.login(key=secret_value_0)
    user = "re-2023"
    project = "DDI_Oct_2024"
    display_name = run_name

    wandb.init(entity=user, project=project, name=display_name)
    print("Seed: ", seed)
    
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size = 128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle=False)
    
    dataloader = {'train':train_loader, 'val':val_loader, 'test': test_loader}
    print("Model", model)
    print(f"Setting seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if model=="MLP":
        model = MLPModel()
    else:
        model = CNNModel()
    print(model)
    dataset_sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset),
        "dev": len(test_dataset)
    }

    
    wandb.finish()
        
def run():
    model = CNNModel()
    dataset_sizes = {
            "train": len(train_direct_dataset),
            "test": len(test_direct_dataset),
            "dev": len(test_direct_dataset)
        }
    # dataloader = {'train':train_loader, 'val':val_loader, 'test': test_loader}
    print("Model", model)
    world_size = torch.cuda.device_count()
#     model, lr, num_epochs, save_model_path, dataloaders, dataset_sizes,  world_size
    mp.spawn(train_model,
             args=(model, 1e-4, 1, 'good.pt', train_direct_dataset, test_direct_dataset, dataset_sizes, world_size),
             nprocs=world_size,
             join=True)

    
def train_model(rank, model, lr, num_epochs, save_model_path, train_direct_dataset, test_direct_dataset, dataset_sizes, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    
    train_loader = DataLoader(train_direct_dataset, batch_size = 128, sampler=DistributedSampler(train_direct_dataset))
    val_loader = DataLoader(test_direct_dataset, batch_size = 128, sampler=DistributedSampler(test_direct_dataset))
    test_loader = DataLoader(test_direct_dataset, batch_size = 128, sampler=DistributedSampler(test_direct_dataset))
    dataloaders = {'train':train_loader, 'val':val_loader, 'test': test_loader}
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

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
            for embed1, embed2, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Progress", total=total_batches):
                embed1 = embed1.to(rank).to(torch.float32)
                embed2 = embed2.to(rank).to(torch.float32)
                labels = labels.to(rank).long()

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(embed1, embed2)
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
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
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
                    print(report)
                    print("Without the negative label:")
                    report = classification_report(
                        y_true=all_labels, 
                        y_pred=all_preds, 
                        labels=[1, 2, 3],  
                        target_names=['New Adverse', 'Antagonism', 'Synergism'],  
                        digits=4,  
                        output_dict=True
                    )
                    print(report)

        if rank == 0:
            torch.save(model.state_dict(), save_model_path)
            
    cleanup()
    return model

if __name__ == "__main__":
    WANDB_KEY="7801339f18c9b00cf55e8f3c250afa3cba1d141b"
    DIRECT_PAIRS_PATH= "/kaggle/input/drugpair-ddi-15oct2024/drug_separate_direct_drugpairdataset.pt"
    UNDIRECT_PAIRS_PATH= "/kaggle/input/drugpair-ddi-15oct2024/drug_separate_undirect_drugpairdataset.pt"
    DRUG_DESC_CSV_PATH= '/kaggle/input/drugbank-ddi/drug_data.csv'
    bert_model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    biomedbert = AutoModel.from_pretrained(bert_model_name).cuda()
    device='cuda'

    drug_description_dict = create_drug_description_dict(DRUG_DESC_CSV_PATH)
    drug_description_dict['DB09368'] = 'Corticotropin zinc hydroxide'
    descriptions = [(id, desc) for id, desc in drug_description_dict.items()]

    # Create a DataLoader for batching
    batch_size = 128  # Adjust based on memory
    data_loader = DataLoader(descriptions, batch_size=batch_size, shuffle=False)

    descs_to_cls_token_embed = {}
    for batch in tqdm(data_loader, desc="Processing batches"):
        ids, descs = batch
        
        inputs = tokenizer(list(descs), return_tensors="pt", padding='max_length', truncation=True, max_length=256).to(device)
        
        with torch.no_grad():
            outputs = biomedbert(**inputs)
        
        # Extract and save the CLS token embedding for each description
        for idx, descs in enumerate(descs):
            descs_to_cls_token_embed[descs] = outputs.last_hidden_state[idx, :, :].to(torch.bfloat16).cpu()

    # FILE
    direct_pairs = torch.load(DIRECT_PAIRS_PATH)
    undirect_pairs = torch.load(UNDIRECT_PAIRS_PATH)
    train_direct_dataset = DrugDataset(direct_pairs.train_dataset, direct_pairs.train_label, descs_to_cls_token_embed)
    test_direct_dataset = DrugDataset(direct_pairs.test_dataset, direct_pairs.test_label, descs_to_cls_token_embed)
    train_indirect_dataset = DrugDataset(undirect_pairs.train_dataset, undirect_pairs.train_label, descs_to_cls_token_embed)
    test_indirect_dataset = DrugDataset(undirect_pairs.test_dataset, undirect_pairs.test_label, descs_to_cls_token_embed)


    n_gpus = torch.cuda.device_count()
    print(f"total GPUs: {n_gpus}")
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run()
    
    