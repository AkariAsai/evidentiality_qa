# Evidentiality-guided Generator
This directory contains the code for:
- Fusion-in-Decoder models
- Evidentiality-guided Generator models

This code is built upon the original [Fusion-in-Decoder](https://github.com/facebookresearch/FiD) implementation.

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**, unlikely to work with a different version)
- [NumPy](http://www.numpy.org/)


## Data

### Download data
We release the DPR retrieved results and the results with our silver evidentiality labels. All of the data can be downloaded from [here](https://drive.google.com/drive/folders/1PA4NEJr3W1JXNvofJYBlTo5nyyGkMqRL?usp=sharing).      
- [`evidentiality_dpr.zip`](https://drive.google.com/file/d/1BnWMB9XS63HPRVq7eWYJ3h4JvVsfr6-5/view?usp=sharing) includes the retrieval results with our newly mined silver evidentiality labels for train sets for each target dataset. For each query, we include top 20 passages. 
- [`eval_dpr.zip`](https://drive.google.com/file/d/1fpmpjHNR0doYdYdS-1298_Jg-V8urPoK/view?usp=sharing) includes the retrieval results for dev / test sets for each target dataset. 

#### About the details of the DPR baselines. 
For Natural Questions and Trivia QA, we run [the Fusion-In-Decoder's script](https://github.com/facebookresearch/FiD/blob/main/get-data.sh) to obtain the DPR retrieval results. Please refer the original repository for more details.            
 
For [FaVIQ-Ambig (Park et al., 2021)](https://arxiv.org/pdf/2107.02153.pdf), we ask the authors of this paper to share the top 100 retrieved passages of their trained retriever. 

For FEVER and Wizard of Wikipedia, we use the [KILT (Petroni et al., 2021)](https://arxiv.org/abs/2009.02252) version, please see [the official repository](https://github.com/facebookresearch/KILT) to download the data as well as the trained retriever model. We had difficulties of running the KILT original code, so we load their official checkpoint and preprocessed Wikipedia tsv file using the [DPR official implementation](https://github.com/facebookresearch/DPR/tree/sewon) to retrieve top passages. 

e.g., 
```
python dense_retriever.py \
    --model_file /path/to/kilt/dpr/model/bert-base-encoder.cp \
    --ctx_file /path/to/kilt/dpr/model/psgs_w100.tsv \
    --qa_file /path/to/input/data/file \
    --encoded_ctx_file "/path/to/embeddings/wikipedia_split/wiki_emb_*" \
    --out_file fever_retrieval_results_test --n-docs 25
```
### Data format
The expected data format is a list of entry examples, where each entry example is a dictionary containing
- `id`: example id, optional
- `question`: question text
- `target`: answer used for model training, if not given, the target is randomly sampled from the 'answers' list
- `answers`: list of answer text for evaluation, also used for training if target is not given
- `ctxs`: a list of passages where each item is a dictionary containing
        - `title`: article title
        - `text`: passage text
        - `has_answer`: evidentiality label. `True` for a positive passage; `False` for a negative passage. 


## Training 

### Fusion-in-Decoder
To train the Fusion-in-Decoder model, please run the command below. 
```
python train_reader.py \
    --use_checkpoint --lr 0.00005 --optim adamw \
    --scheduler linear --weight_decay 0.01 \
    --text_maxlength 250 --per_gpu_batch_size 1 \
    --n_context 2 --total_step 120000 \
    --warmup_step 1000 \
    --train_data /path/to/train/data.json \
    --eval_data /path/to/train/data.json \
    --model_size base --name /model_name/ --accumulation_steps 4 \
    --n_gpus 8 --eval_freq 5000 \
    --answer_maxlength 5

```


### Evidentiality-guided Generator
To train our evidentiality-guided generator model, you first need to obtain the silver evidentiality data. Please see the details in the README.md in the `ranker` directory. 
Once you obtain the silver evidentiality data, you need to change the `has_answer`fields of each context. Then, you can train our evidentiality-guided generator model by running the command below. 

```
python train_reader.py \
    --lr 0.00005 --optim adamw \
    --scheduler linear --weight_decay 0.01 \
    --text_maxlength 250 --per_gpu_batch_size 1 \
    --n_context 20 --total_step 120000 \
    --warmup_step 1000 --train_data /path/to/train/data.json \
    --eval_data /path/to/train/data.json \
    --model_size base --name /model_name/ \
    --accumulation_steps 4 --n_gpus 8 --eval_freq 5000 \
    --joint \
    --answer_maxlength 20
```

## Evaluation
To run the evaluation, please run the command below.

```
python test_reader.py \
    --model_path /path/to/trained/model \
    --eval_data /path/tp/eval/data.json \
    --per_gpu_batch_size 1 --n_context 20 --name /path/to/output/dir \
    --checkpoint_dir checkpoint --n_gpus 1 --write_results 
```