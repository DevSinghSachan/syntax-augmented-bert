
for seed in 40 41 42 43 44; do
    for model_type in "baseline" "m1" "m3"; do
	    for dataset in "conll2005_srl_udv2" "conll2005_srl_udv2_stanza_parses"; do
            export SEED=${seed}
	        export SYNTAX_MODEL_TYPE=${model_type}
	        export DATASET_NAME=${dataset}
	        COMMON_ARGS="--time=15:00:0 --gres=gpu:v100:1 -c 8 --mem=32G --partition=long --export=SEED,SYNTAX_MODEL_TYPE,DATASET_NAME --output=${DATASET_NAME}_${SYNTAX_MODEL_TYPE}_${SEED}.out"
	        sbatch ${COMMON_ARGS} --job-name=${DATASET_NAME}_${SYNTAX_MODEL_TYPE}_${SEED} slurm_job_mila_conll_srl.sh
        done    
    done
done

