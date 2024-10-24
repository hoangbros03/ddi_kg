# ddi_kg

## How to run

First, clone the repo and install the dependencies from `requirements.txt`

### Making the data

We can use the following command to generate the data:

```
mkdir <output_folder>
python3 make_data.py --drug_info_path <The json file holding drug information>  \
--all_pairs_path <The csv file holding all drug pairs> \
--output_folder <output_folder> \
--bert_model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract \
--use_bfloat16 \
--handle_direct \
--discard_missing \
--false_set_limit 10000 \
--false_direct_limit 10000 \
--train_ratio 0.8 \
--seed 42
```

Then it will save into a folder with these files:
```
id2allembedding_cls.pt # Embed [CLS] of modalities of drugs
id2allembedding.pt # Similar but hold all (big file!)
train_dataset.pt # object of DrugDataset holding drugs pairs
test_dataset.pt
```

Remember that to make the dataset work correctly, we must import the DrugDataset from `./data/dataset`, load via `torch.load`, and change the inner variable `id2embedding` from what loaded from `id2embedding` file.