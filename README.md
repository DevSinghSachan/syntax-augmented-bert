
###### Convert from PTB format to UDv2 format

* Install `PyStanfordDependencies` from my fork at: https://github.com/DevSinghSachan/PyStanfordDependencies
  This fork contains changes to use `UniversalDependenciesConverter` class in StanfordCoreNLPv4.0.0.
*


###### Evaluate Stanza Parses Accuracy
*



Syntax-Augmented BERT Models
============================


- module load python/3.6.3
- source ~/.bash_profile

```
export TASK_NAME='tacred'
export GLUE_DIR=${HOME}'/scratch/dataset'
export CACHE_MODEL_PATH=${HOME}'/scratch/.cache/torch/pytorch_transformers/bert'
```

```
python main.py --model_type bert  --model_name_or_path ${CACHE_MODEL_PATH}/bert-base-uncased --task_name $TASK_NAME --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_eval_batch_size 48 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --output_dir $TASK_NAME/ --save_steps 200 --overwrite_output_dir --eval_all_checkpoints --num_train_epochs 10 --do_eval --do_train
```


#### Reproduce results on the CoNLL-2005 SRL task
Download the dataset from here: https://drive.google.com/file/d/1oYK3uskhyrea24KwugXR-56YGlX7mDJC/view?usp=sharing


### Steps
1. Download BERT/RoBERTa model weights from Huggingface by running the command
```
bash download_bert_model.sh
```
This script will create a directory "*hf-bert-weights*" and download the weights of four models. 
Warning: The size of the directory would

2. Create a virtualenv named `pyt` 