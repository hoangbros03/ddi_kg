from collections import defaultdict
import json
import random
import re
import sys
sys.path.append(".")
import logging
import csv

import pandas as pd
import torch
from tqdm import tqdm

# Set up logging, especially in Kaggle
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(stream = sys.stdout, level=logging.INFO)

class DrugPairDataset():
    def __init__(self, data_path, drug_info_path,
                 train_ratio=0.8,
                 type_info=["name", "description", "isomeric_smiles", "formula"],  
                 column_names=['drug1', 'drug2', 'label'], 
                 seed=42,
                 handle_direct=False,
                 false_set_limit=10000,
                 false_direct_limit=10000,
                 discard_missing=True):
        # Path to drug interactions dataset and drug_informations
        self.data_path = data_path
        self.drug_info_path = drug_info_path
        
        # Drug id - information dictionary
        self.drug_info = self.read_drug_info(self.drug_info_path)
        
        # Training ratio
        self.train_ratio= train_ratio
        
        # Type of information using for classification
        self.type_info=type_info
        self.seed= seed
        
        # Number of false samples, only apply to train
        self.false_set_limit = false_set_limit
        self.false_direct_limit = false_direct_limit
       
        # Handle the dataset directly or not
        self.handle_direct = handle_direct
        self.discard_missing = discard_missing

        # Dataset
        self.data = []
        self.label = []
        self.label_set = set()
        self.handle_dataset(data_path, column_names)

    # Reading drug informations from json file
    def read_drug_info(self, drug_info_path):
        with open(drug_info_path, 'r') as json_file:
            drug_info = json.load(json_file) 
        # Add exception
        drug_info['DB09368'] = {
        "name": "Corticotropin zinc hydroxide",
        "description": None,
        "cid": None,
        "canonical_smiles": None,
        "isomeric_smiles": None,
        "formula": None
        }
        return drug_info

    # Check if any field is missing
    def check_informations(self, drug_info_dict, drug_id, informations):
        # Check if the drug_id exists in the dictionary
        if drug_id not in drug_info_dict:
            return False
        
        # If not set discard missing
        if not self.discard_missing:
            return True
        
        # Iterate through the fields and check if any have a None value
        drug_info = drug_info_dict[drug_id]
        for information in informations:
            if information in drug_info and drug_info[information] is None:
                return False
            
        # If no None values were found, return True
        return True

    # Creating the drug dataset
    def handle_dataset(self, path, column_names):
        """
        Receive a csv file as input, including columns: Drug1, Drug2, label, origin_label.
        """
        drug1, drug2, label = column_names
        df = pd.read_csv(path)
        self.drugs_set = set()
        self.pairs_set = set()
        self.undirect_pairs_set = set()

        # Add to set and data
        for i in tqdm(range(len(df[drug1]))):
            drug1_id = df[drug1][i].split("::")[-1]
            drug2_id = df[drug2][i].split("::")[-1]
            
            int_label = df[label][i]
            # Check before add:
            if not self.handle_direct and ((drug1_id, drug2_id) in self.pairs_set or (drug2_id, drug1_id) in self.pairs_set):
                continue
            self.data.append([drug1_id, drug2_id])
            self.pairs_set.add((drug1_id, drug2_id))
            
            if int_label == 'New Adverse' or not self.handle_direct:
                self.undirect_pairs_set.add((drug1_id, drug2_id))

            self.drugs_set.add(drug1_id)
            self.drugs_set.add(drug2_id)
            self.label.append(int_label)
            self.label_set.add(int_label)
        
        # Create a set of drugs with all information
        self.valid_drugs_set = set()
        for drug in self.drugs_set:
            if self.check_informations(self.drug_info, drug, self.type_info):
                self.valid_drugs_set.add(drug)

        self.label_set = {item: index+1 for index, item in enumerate(self.label_set)}
        logging.info(f"The label set is:\n{self.label_set}")
        # logging.info("Data len from csv:", )
        self.label = [self.label_set[i] for i in self.label]
        
        # Add false data
        # false_data = self.get_false_data()
        # false_label = [0] * len(false_data)
        # self.data.extend(false_data)
        # self.label.extend(false_label)
        # self.label_set['No Interaction'] = 0
        
        # Split train test
        self.valid_drugs_set = list(self.valid_drugs_set)
        self.shuffle_list = list(range(len(self.valid_drugs_set)))
        random.shuffle(self.shuffle_list)
        self.train_drug_list = [self.valid_drugs_set[i] for i in self.shuffle_list[:int(len(self.shuffle_list)*self.train_ratio)]]
        self.test_drug_list = [self.valid_drugs_set[i] for i in self.shuffle_list[int(len(self.shuffle_list)*self.train_ratio):]]

        # Save all valid pairs (both drugs have required information) into 3 lists:
        # both in train, one of them in train, none of them in train
        self.both_in_train = []
        self.one_in_train = []
        self.none_in_train = []
        self.both_in_train_label = []
        self.one_in_train_label = []
        self.none_in_train_label = []

        for i in tqdm(range(len(self.data))):
            dr1 = self.data[i][0]
            dr2 = self.data[i][1]
            if self.check_informations(self.drug_info, dr1, self.type_info) and self.check_informations(self.drug_info, dr2, self.type_info):
                if dr1 in self.train_drug_list and dr2 in self.train_drug_list:
                    self.both_in_train.append(self.data[i])
                    self.both_in_train_label.append(self.label[i])

                elif dr1 in self.train_drug_list or dr2 in self.train_drug_list:
                    self.one_in_train.append(self.data[i])
                    self.one_in_train_label.append(self.label[i])

                else:
                    self.none_in_train.append(self.data[i])
                    self.none_in_train_label.append(self.label[i])

        logging.info(f"Both in train: {len(self.both_in_train)}, One in train: {len(self.one_in_train)}, None in train: {len(self.none_in_train)}")
        # Split both_in_train
        # Define the size of the test set for both_in_train to be equal to the smaller set between one_in_train and none_in_train
        test_size = max(len(self.one_in_train), len(self.none_in_train))

        drug_occurrences = defaultdict(int)
        for pair in self.both_in_train:
            dr1, dr2 = pair
            drug_occurrences[dr1] += 1
            drug_occurrences[dr2] += 1

        # Shuffle both_in_train to randomly select pairs for the test set
        self.both_in_train_shuffle_list = list(range(len(self.both_in_train)))
        random.shuffle(self.both_in_train_shuffle_list)

        # Select a part of both_in_train to go into the test set, and keep the rest in the train set
        both_in_train_test = []
        both_in_train_train = []
        both_in_train_test_label = []
        both_in_train_train_label = []

        for i in tqdm(range(len(self.both_in_train))):
            dr1, dr2 = self.both_in_train[i]
            # Ensure that taking this pair for the test set won't completely remove all interactions for dr1 and dr2 in the training set
            if drug_occurrences[dr1] > 1 and drug_occurrences[dr2] > 1:
                if len(both_in_train_test) < test_size:
                    both_in_train_test.append(self.both_in_train[i])
                    both_in_train_test_label.append(self.both_in_train_label[i])
                    
                else:
                    both_in_train_train.append(self.both_in_train[i])
                    both_in_train_train_label.append(self.both_in_train_label[i])
                    drug_occurrences[dr1] -= 1
                    drug_occurrences[dr2] -= 1
            else:
                both_in_train_train.append(self.both_in_train[i])
                both_in_train_train_label.append(self.both_in_train_label[i])

        # Get false data for train
        false_data_train = self.get_false_data(self.train_drug_list, self.false_set_limit) + self.get_false_direct_data(both_in_train_train, self.false_direct_limit)
        false_label_train = [0] * len(false_data_train)

        # Final train. No need to shuffle as Dataloader will do that
        self.train_dataset = both_in_train_train + false_data_train
        self.train_label = both_in_train_train_label + false_label_train

        # Get false data for test
        self.test_dataset = both_in_train_test + self.none_in_train + self.one_in_train
        self.test_label = both_in_train_test_label + self.none_in_train_label + self.one_in_train_label
        false_data_test = self.get_false_data(self.test_drug_list, -1)
        false_label_test = [0] * len(false_data_test)
        self.test_dataset += false_data_test
        self.test_label += false_label_test

        logging.info(f"Training dataset size: {len(self.train_dataset)}, Test dataset size: {len(self.test_dataset)}")
        logging.info("Let the dataloader handle the suffling")

        
    def _check_invalid(self, drug1, drug2):
            if (drug1, drug2) in self.pairs_set:
                return True
            if (drug1, drug2) in self.undirect_pairs_set or (drug2, drug1) in self.undirect_pairs_set:
                return True
            return False
    
    def get_false_data(self, drug_list, sampled_pairs):
        """
        Get sampled false data from drug_list
        Parameters:
        drug_list (list): list of drugs to sample from
        sampled_pairs (int): number of false samples to take. If <= 0, get all possible false samples
        Returns:
        list of lists: [[drug1, drug2], ...]
        """
        false_data = []
        amount = 0
        drugs_amount = len(drug_list)
        drugs_list = list(drug_list)
        random.seed(self.seed)
        if sampled_pairs >0:
            with tqdm(total=sampled_pairs) as pbar:
                while True:
                    idx1, idx2 = random.randint(0, drugs_amount-1), random.randint(0, drugs_amount-1)
                    if self._check_invalid(drugs_list[idx1], drugs_list[idx2]):
                        continue
                    false_data.append([drugs_list[idx1], drugs_list[idx2]])
                    self.pairs_set.add((drugs_list[idx1], drugs_list[idx2]))
                    amount+=1
                    pbar.update(1)
                    if amount>=sampled_pairs:
                        break
        else:
            logging.info("Get all false data as sampled_pairs declared <= 0")
            with tqdm(total=drugs_amount*drugs_amount) as pbar:
                for idx1 in range(drugs_amount):
                    for idx2 in range(drugs_amount):
                        if self._check_invalid(drugs_list[idx1], drugs_list[idx2]):
                            continue
                        false_data.append([drugs_list[idx1], drugs_list[idx2]])
                        self.pairs_set.add((drugs_list[idx1], drugs_list[idx2]))
                        amount+=1
                        pbar.update(1)
        logging.info("False data added: %d" % amount)
        return false_data
    
    def get_false_direct_data(self, pair_drugs_list, sampled_pairs):
        """
        Generates false drug interaction pairs by reversing drug pairs from a given list.

        This function samples a specified number of false drug interaction pairs by reversing
        the order of drug pairs in the provided list. It ensures that the generated pairs
        are not already present in the existing dataset.

        Args:
            pair_drugs_list (list): List of drug pairs to sample from.
            sampled_pairs (int): Number of false samples to generate.

        Returns:
            list: A list containing the generated false drug interaction pairs.
        """
        amount = 0
        false_data = []
        with tqdm(total=sampled_pairs) as pbar:
            while True:
                idx = random.randint(0, len(pair_drugs_list)-1)
                if self._check_invalid(pair_drugs_list[idx][1], pair_drugs_list[idx][0]):
                        continue
                false_data.append([pair_drugs_list[idx][1], pair_drugs_list[idx][0]])
                self.pairs_set.add((pair_drugs_list[idx][1], pair_drugs_list[idx][0]))
                amount+=1
                pbar.update(1)
                if amount>=sampled_pairs:
                    break
        logging.info("False direct data added: %d", amount)
        return false_data
    
def write_to_csv(triples, filename, header=["Drug1", "Interaction", "Drug2"]):
    with open(filename, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        
        # Write the header if needed
        writer.writerow(header)
        
        # Write each triple to the CSV
        for triple in triples:
            writer.writerow(triple)

    logging.info(f"Data written to {filename}")
