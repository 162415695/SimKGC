import os
import glob
import json
import torch
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import predict
import tqdm
from time import time
from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from copy import deepcopy
from typing import List, Tuple
from evaluate import entity_dict, compute_metrics, PredInfo
from doc import Dataset, collate, _convert_is_test_2_true, _convert_is_test_2_false, load_data, Example
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj, \
    concatenate_dict_arrays, generate_random_numbers, copy_checkpoint
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from collections import OrderedDict


def model_load(ckt_path):
    ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
    state_dict = ckt_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_state_dict[k] = v
    return new_state_dict


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        if args.pretrained_ckpt is not None:
            logger.info("读取已有模型权重")
            try:
                self.model.load_state_dict(model_load(ckt_path=args.pretrained_ckpt), strict=True)
                if torch.cuda.is_available():
                    self.model.cuda()
                    self.use_cuda = True
                logger.info('Load model from {} successfully'.format(args.pretrained_ckpt))
            except:
                logger.info("读取失败")
        logger.info(self.model)
        self._setup_training()
        self.extra_batch_size = 0
        self.extra_flag = self.args.add_extra_batch
        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)
        self.extra_batch_limit = args.extra_batch_limit
        if self.extra_batch_limit == -1 or self.extra_batch_limit > len(self.train_loader):
            self.extra_batch_limit = len(self.train_loader) - 1
            logger.info("额外batch上限因为数据量调整为" + str(self.extra_batch_limit))

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)
        # metric_dict = self.eval_epoch(filename)
        metric_dict = self.eval_entity(filename)
        is_best = self.best_metric is None or (metric_dict['hit@1'] > self.best_metric['hit@1'])
        if is_best:
            self.best_metric = metric_dict
        copy_checkpoint(filename, is_best)

    @torch.no_grad()
    def eval_entity(self, epoch) -> Dict:
        self.model.eval()
        _convert_is_test_2_true()
        entity_tensor = self.predict_by_entities(entity_dict.entity_exs)
        forward_metrics = self.eval_single_direction(entity_tensor=entity_tensor, eval_forward=True)
        backward_metrics = self.eval_single_direction(entity_tensor=entity_tensor, eval_forward=False)
        metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
        logger.info('Averaged metrics: {}'.format(metrics))
        _convert_is_test_2_false()
        return metrics

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        total_valid_batch = {i: k for i, k in enumerate(self.valid_loader)}
        model = get_model_obj(self.model)
        candidate_index = [ind for ind in range(len(self.valid_loader))]
        self.model.eval()
        total_tail_id = []
        tail_vector = []
        with torch.no_grad():
            for temp_index in candidate_index:
                indices_to_remove = []
                temp_data = total_valid_batch[temp_index].copy()
                for f in range(len(temp_data['batch_data'])):
                    if temp_data['batch_data'][f].tail_id in total_tail_id:
                        indices_to_remove.append(f)
                    else:
                        total_tail_id.append(temp_data['batch_data'][f].tail_id)
                for key in temp_data:
                    temp_data[key] = np.delete(temp_data[key], indices_to_remove, axis=0)
                temp_data = move_to_cuda(temp_data)
                tail_vector.append(model._encode(model.tail_bert,
                                                 token_ids=temp_data['tail_token_ids'],
                                                 mask=temp_data['tail_mask'],
                                                 token_type_ids=temp_data['tail_token_type_ids']
                                                 ))
        for i, batch_dict in total_valid_batch.items():
            batch_size = len(batch_dict['batch_data'])
            if torch.cuda.is_available():
                tail_vector = move_to_cuda(tail_vector)
                batch_dict = move_to_cuda(batch_dict)
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)
            outputs = model.compute_logits(output_dict=outputs, batch_dict=batch_dict,
                                           extra_tail=tail_vector)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))
        total_train_batch = {i: k for i, k in enumerate(self.train_loader)}
        for i, batch_dict in total_train_batch.items():
            model = get_model_obj(self.model)
            candidate_index = generate_random_numbers(self.extra_batch_size, i, len(total_train_batch))
            self.model.eval()
            total_tail_id = [d.tail_id for d in batch_dict['batch_data']]
            tail_vector = []
            with torch.no_grad():
                for temp_index in candidate_index:
                    indices_to_remove = []
                    temp_data = total_train_batch[temp_index].copy()
                    for f in range(len(temp_data['batch_data'])):
                        if temp_data['batch_data'][f].tail_id in total_tail_id:
                            indices_to_remove.append(f)
                        else:
                            total_tail_id.append(temp_data['batch_data'][f].tail_id)

                    for key in temp_data:
                        temp_data[key] = np.delete(temp_data[key], indices_to_remove, axis=0)

                    temp_data = move_to_cuda(temp_data)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            tail_vector.append(model._encode(model.tail_bert,
                                                             token_ids=temp_data['tail_token_ids'],
                                                             mask=temp_data['tail_mask'],
                                                             token_type_ids=temp_data['tail_token_type_ids']
                                                             ))
                    else:
                        tail_vector.append(model._encode(model.tail_bert,
                                                         token_ids=temp_data['tail_token_ids'],
                                                         mask=temp_data['tail_mask'],
                                                         token_type_ids=temp_data['tail_token_type_ids']
                                                         ))
            if torch.cuda.is_available():
                tail_vector = move_to_cuda(tail_vector)
                batch_dict = move_to_cuda(batch_dict)

            self.model.train()
            batch_size = len(batch_dict['batch_data'])
            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)

            outputs = model.compute_logits(output_dict=outputs, batch_dict=batch_dict,
                                           extra_tail=tail_vector)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            assert logits.size(0) == batch_size
            # head + relation -> tail
            loss = self.criterion(logits, labels)
            # tail -> head + relation

            loss += self.criterion(logits[:, :batch_size].t(), labels)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))

            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
                if self.extra_flag:
                    if acc1 > 90:

                        logger.info("acc1已超过90%,添加额外待预测的尾实体")
                        if self.extra_batch_size == 0 and self.extra_batch_limit != 0:
                            self.extra_batch_size = 1
                            logger.info("尾实体添加成功,当前额外batch数量为" + str(self.extra_batch_size))
                        else:
                            if self.extra_batch_size < self.extra_batch_limit:
                                self.extra_batch_size *= 2
                                if self.extra_batch_size > self.extra_batch_limit:
                                    self.extra_batch_size = self.extra_batch_limit
                                logger.info("尾实体添加成功,当前额外batch数量为" + str(self.extra_batch_size))
                            else:
                                logger.info("尾实体数量已达到预定义上限,修改请参考extra-batch-limit参数")
                                self.extra_flag = False
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.args.task),
            num_workers=1,
            batch_size=max(self.args.batch_size, 512),
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list = [], []
        for idx, batch_dict in enumerate(data_loader):
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.args.task),
            num_workers=2,
            batch_size=max(self.args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)

    @torch.no_grad()
    def eval_single_direction(self,
                              entity_tensor: torch.tensor,
                              eval_forward=True,
                              batch_size=256) -> dict:
        start_time = time()
        examples = load_data(self.args.valid_path, add_forward_triplet=eval_forward,
                             add_backward_triplet=not eval_forward)

        hr_tensor, _ = self.predict_by_examples(examples)
        hr_tensor = hr_tensor.to(entity_tensor.device)
        target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
        logger.info('predict tensor done, compute metrics...')

        topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                    target=target, examples=examples,
                                                                    batch_size=batch_size)
        eval_dir = 'forward' if eval_forward else 'backward'
        logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

        pred_infos = []
        for idx, ex in enumerate(examples):
            cur_topk_scores = topk_scores[idx]
            cur_topk_indices = topk_indices[idx]
            pred_idx = cur_topk_indices[0]
            cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                              for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

            pred_info = PredInfo(head=ex.head, relation=ex.relation,
                                 tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                                 pred_score=round(cur_topk_scores[0], 4),
                                 topk_score_info=json.dumps(cur_score_info),
                                 rank=ranks[idx],
                                 correct=pred_idx == target[idx])
            pred_infos.append(pred_info)

        logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
        return metrics
