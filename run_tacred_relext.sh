#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/tacred/no_deptree_pruning/%x-%j.out
#SBATCH --error=logs/tacred/no_deptree_pruning/%x-%j.err

set -ex
source ${HOME}"/.virtualenvs/pyt/bin/activate"

export SEED=40
export SYNTAX_MODEL_TYPE="late_fusion"
export DATASET_NAME="tacred_revised"
export model="bert-base-cased"
export BERT_WEIGHTS='hf-bert-weights'

# Save the model runs here
export SAVEDIR='checkpoints/2/'
mkdir -p ${SAVEDIR}

# :<<COMMENT
cp 'datasets/'${DATASET_NAME}'.tar.gz' ${SAVEDIR}
cp -r ${BERT_WEIGHTS}/"${model}" ${SAVEDIR}

# Untar your dataset
tar -xvzf ${SAVEDIR}'/'${DATASET_NAME}'.tar.gz' -C ${SAVEDIR}
# COMMENT

export DATA_DIR=${SAVEDIR}
export CACHE_MODEL_PATH=${SAVEDIR}/${model}

# Remove the cached data files
rm -rf ${SAVEDIR}/${DATASET_NAME}/checkpoint-best-model
rm -rf ${SAVEDIR}/${DATASET_NAME}/cached_*

export TASK_NAME="tacred"
python main.py --model_type "syntax_bert_seq" --model_name_or_path ${CACHE_MODEL_PATH} --task_name ${TASK_NAME} \
--data_dir ${DATA_DIR}/${DATASET_NAME} --max_seq_length 512 --per_gpu_eval_batch_size 16 \
--output_dir ${SLURM_TMPDIR}/${DATASET_NAME}/ --save_steps 1000 \
--overwrite_output_dir --num_train_epochs 20 --do_eval --do_train --evaluate_during_training \
--add_masked_ne_tokens --config_name_or_path "config/re/syntaxBERT${SYNTAX_MODEL_TYPE}.json" --per_gpu_train_batch_size 12 \
--gradient_accumulation_steps 3 --wordpiece_aligned_dep_graph --seed ${SEED}

deactivate
