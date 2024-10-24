mkdir output_data
python3 make_data.py --drug_info_path ../data/drug_info.csv --all_pairs_path ../data/all_pairs.csv --output_folder output_data \
--bert_model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --use_bfloat16 --handle_direct --discard_missing \
--false_set_limit 10000 --false_direct_limit 10000 --train_ratio 0.8 --seed 42