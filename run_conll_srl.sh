#!/bin/bash
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=12:00:00                   # The job will run for 8 hours
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -ex
# 1. Create your environement locally
source $HOME'/.virtualenvs/pyt/bin/activate'

export DATASET_NAME='conll2005_srl_udv2'
export SEED=40
export SYNTAX_MODEL_TYPE="m3"
export model='bert-base-cased'
export SCRATCH='/network/tmp1/sachande'

#:<<COMMENT
cp ${SCRATCH}'/dataset/'${DATASET_NAME}'.tar.gz' ${SLURM_TMPDIR}
cp -r ${SCRATCH}'/.cache/torch/pytorch_transformers/bert/'${model} ${SLURM_TMPDIR}

# 3. Eventually unzip your dataset
tar -xvzf ${SLURM_TMPDIR}'/'${DATASET_NAME}'.tar.gz' -C ${SLURM_TMPDIR}
#COMMENT

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
# and look for the dataset into $SLURM_TMPDIR
export DATA_DIR=${SLURM_TMPDIR}
export CACHE_MODEL_PATH=${SLURM_TMPDIR}/${model}

# Remove the cached data files
rm -rf ${SLURM_TMPDIR}/${DATASET_NAME}/checkpoint-best-model
rm -rf $SLURM_TMPDIR/${DATASET_NAME}/cached_*

# Train and Evaluate on WSJ Split
export TASK_NAME='conll2005wsj_srl'
python main.py --model_type "syntax_bert_tok" --model_name_or_path ${CACHE_MODEL_PATH} --task_name ${TASK_NAME} \
--data_dir ${DATA_DIR}/${DATASET_NAME} --max_seq_length 512 --per_gpu_eval_batch_size 32 \
--output_dir ${SLURM_TMPDIR}/${DATASET_NAME}/ --save_steps 2000 \
--overwrite_output_dir --num_train_epochs 20 --do_eval --do_train --evaluate_during_training \
--config_name_or_path "config/srl/syntaxBERT${SYNTAX_MODEL_TYPE}adaptive.json" --per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 --wordpiece_aligned_dep_graph --seed ${SEED}

# Evaluate on Brown Split
export TASK_NAME='conll2005brown_srl'
python main.py --model_type "syntax_bert_tok" --model_name_or_path ${CACHE_MODEL_PATH} --task_name ${TASK_NAME} \
--data_dir ${DATA_DIR}/${DATASET_NAME} --max_seq_length 512 --per_gpu_eval_batch_size 32 \
--output_dir ${SLURM_TMPDIR}/${DATASET_NAME}/ --save_steps 1000 \
--overwrite_output_dir --num_train_epochs 20 --do_eval \
--config_name_or_path "config/srl/syntaxBERT${SYNTAX_MODEL_TYPE}adaptive.json" --per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 --wordpiece_aligned_dep_graph

# 5. Copy whatever you want to save on $SCRATCH
SAVE_PATH="/network/tmp1/singhsad/saved_models/usefulness_syntax/exp1_${DATASET_NAME}_${model}_${SYNTAX_MODEL_TYPE}_${SEED}"
mkdir ${SAVE_PATH}
cp -r ${SLURM_TMPDIR}/${DATASET_NAME} ${SAVE_PATH}

deactivate
