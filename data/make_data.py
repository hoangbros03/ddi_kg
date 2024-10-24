import sys
sys.path.append(".")
import argparse
import logging

import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

from .dataset import DrugDataset, process_formula, get_dict_embedding
from .data_utils import DrugPairDataset

# Set up logging, especially in Kaggle
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(stream = sys.stdout, level=logging.DEBUG)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for drugs using BiomedBERT')
    parser.add_argument('--drug_info_path', type=str, required=True, help='Path to the drug info csv file')
    parser.add_argument('--all_pairs_path', type=str, required=True, help='Path to the all pairs csv file')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder to store the embeddings')
    parser.add_argument('--bert_model', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', help='Bert model to use')
    parser.add_argument('--use_bfloat16', action='store_true', help='Use bfloat16')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--handle_direct', action='store_true', help='Whether to handle the dataset direct or not')
    parser.add_argument('--discard_missing', action='store_true', help='Whether to discard missing data or not')
    parser.add_argument('--false_set_limit', type=int, default=10000, help='Number of false samples for training')
    parser.add_argument('--false_direct_limit', type=int, default=10000, help='Number of false samples for training')

    args = parser.parse_args()

    logging.info("Load bert models")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(args.bert_model).to(device)
    drug_pair_dataset = DrugPairDataset(
        data_path=args.all_pairs_path,
        drug_info_path=args.drug_info_path,
        train_ratio=args.train_ratio,
        type_info=["name", "description", "isomeric_smiles", "formula"],
        column_names=['Drug1', 'Drug2', 'Interaction'],
        seed=args.seed,
        handle_direct=args.handle_direct,
        false_set_limit=args.false_set_limit,
        false_direct_limit=args.false_direct_limit,
        discard_missing=args.discard_missing
    )

    id2allembedding_cls = get_dict_embedding(drug_pair_dataset, tokenizer, model, device, take_cls=True, use_bfloat16=args.use_bfloat16)
    # id2allembedding_full = get_dict_embedding(drug_pair_dataset, tokenizer, model, device, take_cls=False, use_bfloat16=args.use_bfloat16)

    # Don't pass id2allembedding to save the memory. Use it later
    train_dataset = DrugDataset(drug_pair_dataset.train_dataset, drug_pair_dataset.train_label, None)
    test_dataset = DrugDataset(drug_pair_dataset.test_dataset, drug_pair_dataset.test_label, None)

    torch.save(id2allembedding_cls, args.output_folder + "/id2allembedding_cls.pt")
    # torch.save(id2allembedding_full, args.output_folder + "/id2allembedding_full.pt")
    torch.save(train_dataset, args.output_folder + "/train_dataset.pt")
    torch.save(test_dataset, args.output_folder + "/test_dataset.pt")

    logging.info("All processes done")

