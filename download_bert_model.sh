CACHE_DIR=$1
S3_BASEURL='https://s3.amazonaws.com/models.huggingface.co/bert/'
VOCAB_SUFFIX='-vocab.txt'
MODEL_SUFFIX='-pytorch_model.bin'
CONFIG_SUFFIX='-config.json'

models_array=('bert-base-uncased' 'bert-base-cased' 'bert-large-uncased' 'bert-large-cased' 'bert-large-uncased-whole-word-masking' 'bert-large-cased-whole-word-masking')
#models_array=('roberta-base' 'roberta-large')
#models_array=('bert-base-multilingual-cased')

for model in ${models_array[@]};
do
    MODEL_CACHE=${CACHE_DIR}'/'${model}'/'
    mkdir -p ${MODEL_CACHE}
    wget ${S3_BASEURL}${model}${VOCAB_SUFFIX} -O ${MODEL_CACHE}'vocab.txt'
    wget ${S3_BASEURL}${model}${MODEL_SUFFIX} -O ${MODEL_CACHE}'pytorch_model.bin'
    wget ${S3_BASEURL}${model}${CONFIG_SUFFIX} -O ${MODEL_CACHE}'config.json'
done
