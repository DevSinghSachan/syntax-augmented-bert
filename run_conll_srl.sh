#!/bin/bash
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=12:00:00                   # The job will run for 12 hours
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -ex

# Activate your virtual environment locally
source $HOME'/.virtualenvs/pyt/bin/activate'

export DATASET_NAME='conll2005_srl_udv2'
export SEED=40
export SYNTAX_MODEL_TYPE="joint_fusion"
export model='bert-base-cased'
export BERT_WEIGHTS='hf-bert-weights'

# Save the model runs here
export SAVEDIR='checkpoints/1/'
mkdir -p ${SAVEDIR}

# :<<COMMENT
cp 'datasets/'${DATASET_NAME}'.tar.gz' ${SAVEDIR}
cp -r ${BERT_WEIGHTS}/"${model}" ${SAVEDIR}

# Untar your dataset
tar -xvzf ${SAVEDIR}'/'${DATASET_NAME}'.tar.gz' -C ${SAVEDIR}
# COMMENT

# Launch your job, tell it to save the model in $SLURM_TMPDIR and look for the dataset into $SLURM_TMPDIR
export DATA_DIR=${SAVEDIR}
export CACHE_MODEL_PATH=${SAVEDIR}/${model}

# Remove the cached data files
rm -rf ${SAVEDIR}/${DATASET_NAME}/checkpoint-best-model
rm -rf ${SAVEDIR}/${DATASET_NAME}/cached_*

# Train and Evaluate on WSJ Split
export TASK_NAME='conll2005wsj_srl'
python main.py --model_type "syntax_bert_tok" --model_name_or_path ${CACHE_MODEL_PATH} --task_name ${TASK_NAME} \
--data_dir ${DATA_DIR}/${DATASET_NAME} --max_seq_length 512 --per_gpu_eval_batch_size 32 \
--output_dir ${SAVEDIR}/${DATASET_NAME}/ --save_steps 2000 \
--overwrite_output_dir --num_train_epochs 20 --do_eval --do_train --evaluate_during_training \
--config_name_or_path "config/srl/bert-base/${SYNTAX_MODEL_TYPE}.json" --per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 --wordpiece_aligned_dep_graph --seed ${SEED}

# Evaluate on Brown Split
export TASK_NAME='conll2005brown_srl'
python main.py --model_type "syntax_bert_tok" --model_name_or_path ${CACHE_MODEL_PATH} --task_name ${TASK_NAME} \
--data_dir ${DATA_DIR}/${DATASET_NAME} --max_seq_length 512 --per_gpu_eval_batch_size 32 \
--output_dir ${SLURM_TMPDIR}/${DATASET_NAME}/ --save_steps 1000 \
--overwrite_output_dir --num_train_epochs 20 --do_eval \
--config_name_or_path "config/srl/bert-base/${SYNTAX_MODEL_TYPE}.json" --per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 --wordpiece_aligned_dep_graph

deactivate
