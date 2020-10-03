from argparse import ArgumentParser

from utils.utils import processors


def get_args():
    parser = ArgumentParser(description='Finetuning Syntax-Augmented BERT/RoBERTa models')

    group = parser.add_argument_group('--bert_options')
    # Required parameters
    group.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    group.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type")
    group.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    group.add_argument("--config_name_or_path", default=None, type=str, required=True,
                       help="Path to pre-trained config or shortcut name selected in the list")
    group.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    group.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    group.add_argument("--no_pretrained", action='store_true',
                        help="Whether to use pretrained BERT model weights, else use random initialization")
    group.add_argument("--use_swa", action='store_true',
                       help="Whether to apply Stochastic Weight Averaging (SWA)")
    group.add_argument("--add_masked_ne_tokens", action='store_true',
                        help="Whether to include masked Named Entity tokens as additional vocabulary in TACRED RE task")
    group.add_argument('--wordpiece_aligned_dep_graph', action='store_true',
                       help='align the dependency graph according to wordpiece tokens')
    group.add_argument('--sample_rate', type=float, default= 1.0,
                       help='align the dependency graph according to wordpiece tokens')

    # Other parameters
    group.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    group.add_argument("--update_config_str", type=str, default=None,
                       help="dict in str form to update the config parameters")
    group.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    group.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    group.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    group.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    group.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    group.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    group.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    group.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    group.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    group.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    group.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    group.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    group.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    group.add_argument('--write_eval_results', action='store_true',
                       help="Write evaluation results per example.")
    group.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    group.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    group.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    group.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    group.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    group.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    group.add_argument("--local_rank", type=int, default=-1,
                       help="For distributed training: local_rank")

    group = parser.add_argument_group('--fp16_options')
    group.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    group.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()
    return args
