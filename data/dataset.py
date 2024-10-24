import re

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class DrugDataset(Dataset):
    def __init__(self, data, labels, type_info, id2allembedding):
        super(DrugDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.type_info = type_info
        self.id2allembedding = id2allembedding
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract the label and the pair (name1, name2) from the DataFrame
        
        dr1 = self.data[idx][0]
        dr2 = self.data[idx][1]
        label = self.labels[idx]

        embed1, embed2 = list(), list()

        for info in self.type_info:
            embed1.append(self.id2allembedding[dr1][info])
            embed2.append(self.id2allembedding[dr2][info])

        embed = tuple(embed1 + embed2 + label)
        
        return embed
    
def process_formula(string):   
    """
    Processes a chemical formula string by expanding element abbreviations to full names
    and ensuring numeric subscripts are explicitly represented.

    The function takes a chemical formula string, identifies element symbols and their
    respective quantities, replaces element abbreviations with their full names, and
    ensures each element has an explicit numeric subscript. If a quantity is not specified
    for an element, it defaults to '1'.

    Args:
        string (str): The chemical formula string to process.

    Returns:
        str: A processed string with full element names and explicit quantities.
    """
    map_abrv_dict = {
        'Ag': 'Silver',
        'Al': 'Aluminum',
        'As': 'Arsenic',
        'Au': 'Gold',
        'B': 'Boron',
        'Ba': 'Barium',
        'Bi': 'Bismuth',
        'Br': 'Bromine',
        'C': 'Carbon',
        'Ca': 'Calcium',
        'Cl': 'Chlorine',
        'Co': 'Cobalt',
        'Cu': 'Copper',
        'F': 'Fluorine',
        'Fe': 'Iron',
        'H': 'Hydrogen',
        'Hg': 'Mercury',
        'I': 'Iodine',
        'K': 'Potassium',
        'Li': 'Lithium',
        'Mg': 'Magnesium',
        'N': 'Nitrogen',
        'Na': 'Sodium',
        'O': 'Oxygen',
        'P': 'Phosphorus',
        'Pt': 'Platinum',
        'S': 'Sulfur',
        'Se': 'Selenium',
        'Si': 'Silicon',
        'Tc': 'Technetium',
        'Ti': 'Titanium',
        'Zn': 'Zinc'
        }

    char_list = [c for c in string]
    char_list_new = list()
        
    tmp = ''
    for i in range(len(char_list)):
        if re.match('[0-9]', char_list[i]) and re.match('[0-9]', char_list[i-1]):
            tmp += char_list[i]
        elif re.match('[0-9]', char_list[i]):
            char_list_new.append(tmp)
            tmp = char_list[i]
        elif re.match('[a-z]', char_list[i]):
            tmp += char_list[i]
        elif re.match('[A-Z]', char_list[i]):
            char_list_new.append(tmp)
            tmp = char_list[i]
        
    char_list_new.append(tmp)
    char_list_new = char_list_new[1:]
    
    char_list_return = list()
    for i in range(len(char_list_new)):
        if i == len(char_list_new) - 1:
            if re.match('[a-zA-Z]+', char_list_new[i]):
                char_list_return.append(char_list_new[i])
                char_list_return.append('1')
            else:
                char_list_return.append(char_list_new[i])
        elif re.match('[a-zA-Z]+', char_list_new[i]) and re.match('[a-zA-Z]+', char_list_new[i+1]):
            char_list_return.append(char_list_new[i])
            char_list_return.append('1')
        else:
            char_list_return.append(char_list_new[i])

    for i in range(len(char_list_return)):
        if char_list_return[i] in map_abrv_dict.keys():
            char_list_return[i] = map_abrv_dict[char_list_return[i]]
            
    return ' '.join(char_list_return)


def get_dict_embedding(drugpairdataset, tokenizer, model, device, take_cls=False, use_bfloat16=False):
    def get_embedding(tokenizer, model, text, max_length, use_bfloat16=False):
        """
        Generates an embedding for the given text using a pre-trained BiomedBERT model.

        Args:
            text (str): The input text to be converted into an embedding.
            max_length (int): The maximum length for the tokenized input sequence.

        Returns:
            torch.Tensor: The embedding of the input text, represented as a PyTorch tensor.
        """
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.to("cpu")
        if take_cls:
            embedding = embedding[:, 0, :]
        if use_bfloat16:
            embedding = embedding.to(torch.bfloat16).cpu()
        return embedding
    
    NAME_MAX_LENGTH = 64
    DESCRIPTION_MAX_LENGTH = 512
    SMILE_MAX_LENGTH=128
    FORMULA_MAX_LENGTH=64

    id2allembedding = {}
    for id, info in tqdm(drugpairdataset.drug_info.items(), desc="Processing drugs", total=len(drugpairdataset.drug_info)):
        # Skip if any required field is None
        info = {key: value if value is not None else "" for key, value in info.items()}
        
        # Tokenize and get embeddings
        name_embedding = get_embedding(tokenizer, model, info['name'], max_length=NAME_MAX_LENGTH, use_bfloat16=use_bfloat16)
        description_embedding = get_embedding(tokenizer, model, info['description'], max_length=DESCRIPTION_MAX_LENGTH, use_bfloat16=use_bfloat16)
        smiles_embedding = get_embedding(tokenizer, model, info['isomeric_smiles'], max_length=SMILE_MAX_LENGTH, use_bfloat16=use_bfloat16)
        formula_embedding = get_embedding(tokenizer, model, process_formula(info['formula']), max_length=FORMULA_MAX_LENGTH, use_bfloat16=use_bfloat16)

        # Store embeddings in id2allembedding
        id2allembedding[id] = {
            'name': name_embedding,
            'description': description_embedding,
            'isomeric_smiles': smiles_embedding,
            'formula': formula_embedding
        }
    return id2allembedding

