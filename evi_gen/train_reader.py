# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from tqdm import tqdm

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    device = opt.device
    print("using device: {0}".format(device))
    while step < opt.total_steps:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader)):
            step += 1
            # joint model
            if opt.joint is True:
                (idx, labels, _, has_answers, _, context_ids, context_mask) = batch
                train_loss = model(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    labels=labels.cuda(), 
                    class_labels=has_answers.cuda()
                )[0]
            # original FiD
            else:
                (idx, labels, _, context_ids, context_mask) = batch
                train_loss = model(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    labels=labels.cuda()
                )[0]
                    
            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    curr_loss = 0
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    class_match = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if opt.joint is True:
                (idx, labels, _, has_answers, _, context_ids, context_mask) = batch
            else:
                (idx, _, _, context_ids, context_mask) = batch

            if opt.joint is True:
                model.encoder.n_passages = context_ids.size(1)
                class_context_ids = context_ids.view(context_ids.size(0), -1)
                class_context_mask = context_mask.view(context_mask.size(0), -1)
                class_logits = model(
                    input_ids=class_context_ids.cuda(),
                    attention_mask=class_context_mask.cuda(),
                    labels=labels.cuda(), 
                    class_labels=has_answers.cuda()
                )[-1]

                positive_logits = class_logits[:,0,1465]
                negative_logits = class_logits[:,0,2841]
                pos_scores = [float(pos) for pos in positive_logits]
                neg_scores = [float(neg) for neg in negative_logits]

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=opt.answer_maxlength_eval
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                example = dataset.data[idx[k]]
                contexts = example['ctxs'][:opt.n_context]
                if opt.joint is True:
                    has_answer = ["positive" if c["has_answer"]
                                is True else "negative" for c in contexts]
                if opt.metric == "em":
                    score = src.evaluation.ems(ans, gold)
                elif opt.metric == "f1":
                    score = src.util._metric_max_over_ground_truths(src.util._f1_score, ans, gold)
                else:
                    raise NotImplementedError

                total += 1
                exactmatch.append(score)
                if opt.joint is True:
                    score_dict = {}
                    for j, gt in enumerate(has_answer):
                        pos_score = pos_scores[j]
                        neg_score = neg_scores[j]
                        score_dict[j] = pos_score 
                    sorted_score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
                    positive_passages = [idx for idx, pas in enumerate(has_answer) if pas == "positive"]
                    if len(positive_passages) == 0:
                        continue
                    # check how many of the top 10 passages are actually positive
                    top_10_passages = list(sorted_score_dict.keys())[:10] + [0]
                    class_score = len(set(top_10_passages).intersection(set(positive_passages))) > 0
                    class_match.append(class_score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    if opt.joint is True:
        print("prediction accuracy:{0} / {1} = {2}".format(sum(class_match), len(class_match), sum(class_match)/ len(class_match)))
    return exactmatch

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    # FIXME: temporarily comment off to use default DP
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    print(opt.device)
    print(opt.is_main)

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    # if opt.model_name is None:
    #     trained_model_dir = None
    # else:
    #     trained_model_dir = opt.model_name
    model_class = src.model.FiDT5
    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    if opt.joint is True:
        collator = src.data.CollatorJoint(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    else:
        collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    if opt.joint is True:
        train_dataset = src.data.DatasetJoint(train_examples, opt.n_context)
    else:
        train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    if opt.joint is True:
        eval_dataset = src.data.DatasetJoint(eval_examples, opt.n_context)
    else:
        eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        if opt.joint is True:
            model = src.model.FiDT5Joint(t5.config)
        else:
            model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        # initialize the classification decoder.
        if opt.joint is True:
            if opt.decoder_dir is not None:
                # load already trained decoder model's output
                trained_t5 = transformers.T5ForConditionalGeneration.from_pretrained(opt.decoder_dir)
                model.set_decoder_weights(trained_t5.state_dict())
            else:
                model.set_decoder_weights(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )