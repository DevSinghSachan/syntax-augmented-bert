CACHE_DIR="hf-bert-weights"
S3_BASEURL='https://s3.amazonaws.com/models.huggingface.co/bert/'
VOCAB_SUFFIX='-vocab.txt'
MODEL_SUFFIX='-pytorch_model.bin'
CONFIG_SUFFIX='-config.json'

# Create a directory to save the pre-trained BERT/RoBERTa model weights from huggingface
mkdir ${CACHE_DIR}

# Download vocab, weights, and config files for these models

# BERT Models
models_array=('bert-base-cased' 'bert-large-cased' 'bert-large-cased-whole-word-masking')

for model in ${models_array[@]};
do
    MODEL_CACHE=${CACHE_DIR}'/'${model}'/'
    mkdir -p ${MODEL_CACHE}
    wget ${S3_BASEURL}${model}${VOCAB_SUFFIX} -O ${MODEL_CACHE}'vocab.txt'
    wget ${S3_BASEURL}${model}${MODEL_SUFFIX} -O ${MODEL_CACHE}'pytorch_model.bin'
    wget ${S3_BASEURL}${model}${CONFIG_SUFFIX} -O ${MODEL_CACHE}'config.json'
done

# RoBERTa Models
VOCAB_SUFFIX='-vocab.json'
MERGES_SUFFIX='-merges.txt'
models_array=('roberta-large')

for model in ${models_array[@]};
do
    MODEL_CACHE=${CACHE_DIR}'/'${model}'/'
    mkdir -p ${MODEL_CACHE}
    wget ${S3_BASEURL}${model}${VOCAB_SUFFIX} -O ${MODEL_CACHE}'vocab.json'
    wget ${S3_BASEURL}${model}${MERGES_SUFFIX} -O ${MODEL_CACHE}'merges.txt'
    wget ${S3_BASEURL}${model}${MODEL_SUFFIX} -O ${MODEL_CACHE}'pytorch_model.bin'
    wget ${S3_BASEURL}${model}${CONFIG_SUFFIX} -O ${MODEL_CACHE}'config.json'
done
