# Silver Evidentiality Mining 

This repository contains code for evidentiality labeling model. We also include some scripts to obtain silver evidentiality data.

## Overview 
To mine the silver evidentiality labels, you have to follow the steps listed below: 
1. Train base generator model 
2. Create the input data for leave-one-out generation 
3. Run the base generator on the leave-one-out input data
4. Run a mapping script to collect positive and negative passages from leave-one-out prediction
5. Training an evidentiality labeling model
6. Run evidentiality predictions

## Detailed instructions
### 1. Train base generator model 
First you need to train the base generator model (FiD) using the original training data (x,^y, P). Please see the details in README of the `evi_gen`.

### 2. Create the input data for leave-one-out generation 

```sh
python scripts/create_leave_one_out_data.py \
    --input_fp /path/to/original/train/data.json --top_k 10 --max 20 \
    --output_fp /path/to/loo/input/data.json --more
```

### 3. Run the base generator on the leave-one-out input data

```sh
cd ../evi_gen
python test_reader.py \
    --model_path /path/to/model/file \
    --eval_data  /path/to/loo/input/data.json \
    --n_context 9 --name /path/to/prediction/directory/name \
    --checkpoint_dir checkpoint --n_gpus 1 --write_results
```
Please set `--metric f1` for Wizard of Wikipedia. The final prediction will be at`checkpoint/path/to/prediction/directory/name/final_results.txt`, where each row presents the prediction result with the input id. separated by a tab (e.g., `1\tNew York`)/ 

### 4. Run a mapping script to collect positive and negative passages from leave-one-out prediction
```sh
python3 scripts/create_train_data_from_hold_out.py \
    --input_fp  /path/to/loo/input/data.json \
    --final_preds checkpoint//path/to/prediction/directory/name/final_results.txt \
    --output_dir evi_train_data_dir --top_k 10 
```
If you have another training data (E.g., training data created by Natural Questions gold passages), please add the `--sup_fp /supplementary/train/data.json `.

Please use [`create_train_data_from_hold_out_fever.py`](scripts/create_train_data_from_hold_out_fever.py) for classification tasks such as FEVER or FaVIQ, and [`create_train_data_from_hold_out_wow.py`](scripts/create_train_data_from_hold_out_wow.py) for dialogue.


### 5. Training an evidentiality labeling model
After you collect new positive and negative passages, please train an evidentiality labeling model by using the command below:

```sh
python run_ranker.py \
    --model_name_or_path roberta-base \
    --do_eval --do_train --train_file evi_train_data_dir/train.csv \
    --validation_file /path/to/validation/file --max_seq_length 350  \
    --per_device_train_batch_size 12   --learning_rate 2e-5  \
    --num_train_epochs 7  \
    --output_dir /output/path/name --save_steps 5000
```

To run evidentiality prediction, use the same `run_ranker.py` and path `` 

### 6. Run evidentiality predictions

You first need to convert a retrieval result file (in the DPR output format)

```sh
python3 scripts/convert_dpr_to_ranker_data.py \
    --test_fp /path/to/retrieval/results/for/test/set \
    --top_k 20 --output_dir /path/to/output/dir
```
At `/path/to/output/dir/test.csv`, a tsv file, where each line presents a pair of a query and a passage, will be created. Then you run the ranker model prediction as follows: 

```sh
python run_ranker.py \
    --model_name_or_path /output/path/name \
    --do_predict --test_file /path/to/original/train/data \
    --max_seq_length 350  \
    --output_dir /output/path/name --save_steps 5000
```
Each line of the output txt file shows indicates if the passage corresponds to each input line is `positive` or `negative`.