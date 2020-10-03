# coding=utf-8

from __future__ import absolute_import, division, print_function

import logging
import os
import random

import jsonlines
import numpy as np
import torch
from torch.utils.data import (RandomSampler, SequentialSampler, WeightedRandomSampler)
from tensorboardX import SummaryWriter
import torchcontrib
from tqdm import tqdm, trange

from utils.optimizers import Optimizer
from pytorch_transformers import (BertConfig, BertTokenizer, RobertaTokenizer,
                                  BertForSequenceClassification)

from model import (SyntaxBertForSequenceClassification, SyntaxBertForTokenClassification,
                   SyntaxBertConfig, GNNClassifier,
                   SyntaxRobertaForTokenClassification, SyntaxRobertaConfig)
from utils.utils import (output_modes, processors, scorer)
from utils.loader import FeaturizedDataset, FeaturizedDataLoader
from utils import constant
from utils.update_config import update_config_file
from opt import get_args

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'syntax_bert_seq': (SyntaxBertConfig, SyntaxBertForSequenceClassification, BertTokenizer),
    'syntax_bert_tok': (SyntaxBertConfig, SyntaxBertForTokenClassification, BertTokenizer),
    'gcn': (SyntaxBertConfig, GNNClassifier, BertTokenizer),
    'syntax_roberta_tok': (SyntaxRobertaConfig, SyntaxRobertaForTokenClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, config, loading_info=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    print(model)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = FeaturizedDataLoader(train_dataset,
                                            args,
                                            eval=False,
                                            batch_size=args.train_batch_size,
                                            sampler=train_sampler)

    def check_no_decay(var_name):
        no_decay = ['bias', 'LayerNorm.weight']
        for param in no_decay:
            if param in var_name:
                return True
        return False

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    n_params = sum([p.nelement() for p in model.parameters()])
    print(f'* number of parameters: {n_params}')

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer, scheduler = Optimizer()(model.named_parameters(),
                                       config.optimizer)
    scheduler.fit(t_total)
    if args.use_swa:
        swa_optimizer = torchcontrib.optim.SWA(optimizer,
                                               swa_start=1,
                                               swa_freq=len(train_dataloader))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps *
                (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    logging_loss, max_score = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for epoch in train_iterator:
        tr_loss = 0.0
        grad_norm = 0.
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        score_cls, score_map = scorer[args.task_name]
        score = score_cls(score_map)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            dict_ = model(**batch)
            loss, preds = dict_['loss'], dict_['predict']
            score.update(preds,
                         batch['labels'],
                         batch['verb_index'],
                         batch['input_tokens'],
                         training=True)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)

            grad_norm += gnorm
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                if args.use_swa:
                    swa_optimizer.step()
                else:
                    optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar('gradient_norm', gnorm, global_step)
                    logging_loss = tr_loss
                    logger.info('training loss = {} | global step = {}'.format(tr_loss/ (step + 1), global_step))
                    logger.info("Training Scores")
                    score.get_stats(training=True)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, split='dev')
                        eval_f1 = results['f1-measure-overall']
                        eval_loss = results['eval_loss']
                        tb_writer.add_scalar('eval_F1', eval_f1, global_step)
                        logger.info('eval loss = {} | best dev F1 = {} | global step = {}'.format(eval_loss, max_score, global_step))

                        if eval_f1 > max_score:
                            max_score = eval_f1
                            output_dir = os.path.join(args.output_dir, f'checkpoint-best-model')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info(f'New Best F1 Score!! Saving best model checkpoint to {output_dir}')
                        logger.info(f'Average Gradient Norm at step {global_step}: {grad_norm / (step + 1)}')

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.use_swa:
        swa_optimizer.swap_swa_sgd()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix='', split='dev'):
    eval_task = args.task_name
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args,
                                           eval_task,
                                           tokenizer,
                                           split=split)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = FeaturizedDataLoader(eval_dataset,
                                           args,
                                           eval=True,
                                           batch_size=args.eval_batch_size,
                                           sampler=eval_sampler)

    logger.info("***** Running evaluation {} / Num Examples {} *****".format(prefix, len(eval_dataset)))
    eval_loss = 0.0
    nb_eval_steps = 0
    score_cls, score_map = scorer[args.task_name]
    score = score_cls(score_map)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        guid = batch.pop("guid")
        with torch.no_grad():
            dict_ = model(**batch)
            loss, preds = dict_['loss'], dict_['predict']
            eval_loss += loss.mean().item()
        nb_eval_steps += 1
        score.update(preds,
                     batch['labels'],
                     batch['verb_index'],
                     batch['input_tokens'],
                     guid)

    # Write the Overall eval results to a file
    if args.write_eval_results:
        with jsonlines.open(args.output_dir + f'{args.task_name}_{split}_eval_overall.jsonl', 'w') as writer:
            for dict_ in score._overall:
                writer.write(dict_)

    eval_loss = eval_loss / nb_eval_steps
    logger.info("Evaluation Scores")
    results = score.get_stats()
    results['eval_loss'] = eval_loss
    return results


def load_and_cache_examples(args, task, tokenizer, split='train'):
    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(split,
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split('/'))).pop(),
                                                                                   str(args.max_seq_length),
                                                                                   str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        dataset = FeaturizedDataset(torch.load(cached_features_file), cached_features=True)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_map = processor.get_labels()
        if split == 'train':
            examples = processor.get_train_examples(args.data_dir)
            # Adding Special relation masking tokens for TACRED dataset
            if task == 'tacred':
                class_sample_count = torch.FloatTensor([processor.class_weight[l] for l in label_map])
                args.weight = class_sample_count / class_sample_count.sum()
        elif split == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        dataset = FeaturizedDataset(examples,
                                    args,
                                    tokenizer,
                                    label_map,
                                    cls_token_segment_id=0,
                                    pad_token_segment_id=0)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(dataset.data, cached_features_file)

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if args.local_rank == 0:
        torch.distributed.barrier()

    return dataset


def main():
    args = get_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_map = processor.get_labels()
    args.num_labels = len(label_map)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    if args.add_masked_ne_tokens and args.task_name == 'tacred':
        special_tokens_dict = {'additional_special_tokens': list(constant.TACRED_SPECIAL_ENTITY_SET)}
        tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Adding {len(constant.TACRED_SPECIAL_ENTITY_SET)} new tokens to vocabulary correspondng to Entity Masking")

    if args.update_config_str:
        args.config_name_or_path = update_config_file(args.config_name_or_path,
                                                      args.update_config_str)

    config = config_class.from_pretrained(args.config_name_or_path,
                                          num_labels=args.num_labels,
                                          finetuning_task=args.task_name)

    config.label_map = label_map
    if not args.no_pretrained:
        model, loading_info = model_class.from_pretrained(args.model_name_or_path,
                                                          from_tf=bool('.ckpt' in args.model_name_or_path),
                                                          config=config,
                                                          output_loading_info=True)
    else:
        model = model_class(config=config)
        loading_info = None

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args,
                                                args.task_name,
                                                tokenizer,
                                                split='train')
        # Initializing the bias of the classifier layer in proportion to number of class examples respectively,
        # model.classifier.bias.data = torch.Tensor(constant.TACRED_CLASS_WEIGHTS).to(args.device)

        # Resize token embeddings in case extra vocabulary has been added
        model.resize_token_embeddings(len(tokenizer))
        global_step, tr_loss = train(args,
                                     train_dataset,
                                     model,
                                     tokenizer,
                                     config,
                                     loading_info)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluate the best model on Test set
    if not args.use_swa:
        checkpoint = os.path.join(args.output_dir, 'checkpoint-best-model')
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)

    result = evaluate(args,
                      model,
                      tokenizer,
                      split='dev')
    fstr_dev = f"Best Dev: P: {result['precision-overall'] * 100:.2f} | R: {result['recall-overall'] * 100:.2f} | F1 {result['f1-measure-overall'] * 100:.2f}"
    print(fstr_dev + '\n')

    result = evaluate(args,
                      model,
                      tokenizer,
                      split='test')
    fstr_test = f"Test: P: {result['precision-overall'] * 100:.2f} | R: {result['recall-overall'] * 100:.2f} | F1 {result['f1-measure-overall'] * 100:.2f}"
    print(fstr_test + '\n')

    with open(args.output_dir + f'{args.task_name}_results.txt', 'w') as fp:
        fp.write(fstr_dev)
        fp.write(fstr_test)


if __name__ == "__main__":
    main()
