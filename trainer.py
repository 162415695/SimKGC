import os
import glob
import json
import torch
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import predict
import tqdm
from time import time
from triplet_mask import construct_mask, construct_mask_extra_batch, construct_n_hop_mask
from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from copy import deepcopy
from typing import List, Tuple
from evaluate import entity_dict, compute_metrics, PredInfo, _setup_entity_dict, compute_metrics1
from doc import Dataset, collate, _convert_is_test_2_true, _convert_is_test_2_false, load_data, Example, \
    _concat_name_desc
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj, \
    concatenate_dict_arrays, generate_random_numbers, copy_checkpoint
from metric import accuracy, new_accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from collections import OrderedDict

entity_dict = _setup_entity_dict()


def model_load(ckt_path):
    ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
    state_dict = ckt_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_state_dict[k] = v
    return new_state_dict


"""Sparsemax activation function.
Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SigmoidBCELoss(nn.Module):
    """Sigmoid Binary Cross Entropy
    """

    def __init__(self, weight=1.0, reduction='mean'):
        super(SigmoidBCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.m = nn.Sigmoid()
        self.loss = nn.BCELoss(weight=torch.tensor([self.weight]), reduction=self.reduction)

    def forward(self, logits, labels):
        one_hot_labels = F.one_hot(labels, num_classes=logits.shape[-1]).float()
        output = self.loss(self.m(logits), one_hot_labels)
        return output


class TopKSoftmax(nn.Module):
    def __init__(self, k=10, dim=-1):
        super(TopKSoftmax, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, logits):
        # 获取前K大的值及其索引
        topk_vals, topk_indices = torch.topk(logits, self.k, dim=self.dim)

        # 对前K大的值做Softmax
        topk_softmax = F.sigmoid(topk_vals)

        # 构建与logits形状相同的全0矩阵
        softmax_output = torch.zeros_like(logits)

        # 将Softmax后的值放回相应位置
        softmax_output.scatter_(self.dim, topk_indices, topk_softmax)

        return softmax_output


class SigmoidBCELoss(nn.Module):
    """Sigmoid Binary Cross Entropy
    """

    def __init__(self, weight=1.0, reduction='mean'):
        super(SigmoidBCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.m = nn.Sigmoid()
        self.loss = nn.BCELoss(weight=torch.tensor([self.weight]), reduction=self.reduction)

    def forward(self, logits, labels):
        one_hot_labels = F.one_hot(labels, num_classes=logits.shape[-1]).float()
        output = self.loss(self.m(logits), one_hot_labels)
        return output


class SparsemaxBCELoss(nn.Module):
    """Sparsemax Binary Cross Entropy Loss"""

    def __init__(self, weight=1.0, reduction='mean'):
        super(SparsemaxBCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.topksoft = TopKSoftmax()
        self.loss_fn = nn.BCELoss(weight=torch.tensor([self.weight]), reduction=self.reduction)

    def forward(self, logits, labels):
        # Apply Sparsemax activation
        probs = self.topksoft(logits)

        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes=logits.shape[-1]).float()

        # Compute BCE loss
        loss = self.loss_fn(probs, one_hot_labels)
        return loss


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
                self.model.load_state_dict(model_load(ckt_path=args.pretrained_ckpt), strict=False)
                if torch.cuda.is_available():
                    self.model.cuda()
                    self.use_cuda = True
                logger.info('Load model from {} successfully'.format(args.pretrained_ckpt))
            except Exception as e:
                logger.info("读取失败")
                logger.info(e)
        logger.info(self.model)
        self._setup_training()
        if not args.add_extra_batch:
            self.extra_batch_size = args.extra_batch_limit
        else:
            self.extra_batch_size = 0
        self.extra_flag = self.args.add_extra_batch
        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
        # self.criterion2 = SigmoidBCELoss(reduction='mean').cuda()
        self.criterion2 =  nn.BCEWithLogitsLoss(reduction='mean').cuda()
        tail_bert_params = {id(param): param for param in self.model.module.tail_bert.parameters() if
                            param.requires_grad}

        # 然后，从model的所有参数中排除tail_bert的参数
        params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in tail_bert_params]
        '''
        self.optimizer = AdamW([
            {'params': params,
             'lr': args.lr},  # fc1层的学习率
            {'params': tail_bert_params.values(), 'lr': args.lr /(1+args.extra_batch_limit)}
            # 其他层的学习率
        ], lr=args.lr, weight_decay=args.weight_decay)
        print(self.optimizer)
        '''
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        self.train_steps = num_training_steps
        self.current_steps = 0
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
        epoch = 0

        #self._run_eval(epoch=epoch, extra_batch_num=self.extra_batch_size)
        while epoch < self.args.epochs:
            # train for one epoch
            extra_flag = self.train_epoch(epoch)
            if extra_flag:
                logger.info('已扩大batch,重新进行训练')
                epoch = 0  # 重置为0重新开始
            else:
                epoch += 1  # 继续到下一个epoch
                self._run_eval(epoch=epoch, extra_batch_num=self.extra_batch_size)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0, extra_batch_num=0):
        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        if extra_batch_num > 0:
            filename = '{}/checkpoint_epoch{}_extra_batch{}.mdl'.format(self.args.model_dir, epoch, extra_batch_num)

        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)
        metric_dict = self.eval_entity(epoch)
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

    def reset_learning_rate(self, total_steps):
        # 重置优化器的学习率
        if self.current_steps <= self.args.warmup:
            new_lr = self.args.lr
        else:
            new_lr = self.scheduler.get_last_lr()[0]
        self.current_steps = 0
        self.args.lr = new_lr
        logger.info(new_lr)
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=new_lr,
                               weight_decay=self.args.weight_decay)
        warmup_steps = min(self.args.warmup, total_steps // 10)
        if self.args.lr_scheduler == 'linear':
            # 重新创建调度器
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.args.lr_scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

    def train_epoch(self, epoch):


        if self.extra_flag:
            prefix = "Epoch: [{}],extra_batch:[{}]".format(epoch, self.extra_batch_size)
        else:
            prefix = "Epoch: [{}]".format(epoch)
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix=prefix)
        if self.args.add_discriminator:
            prefix = "Epoch: [{}],discriminator: ".format(epoch)
            losses_dis=AverageMeter('Loss', ':.4')
            top1_dis=AverageMeter('Acc@1', ':6.2f')
            progress_dis = ProgressMeter(
                len(self.train_loader),
                [losses_dis,top1_dis],
                prefix=prefix
            )
        total_train_batch = {i: k for i, k in enumerate(self.train_loader)}
        for i, batch_dict in total_train_batch.items():
            self.current_steps += 1
            model = get_model_obj(self.model)
            candidate_index = generate_random_numbers(self.extra_batch_size, i, len(total_train_batch))
            self.model.eval()
            total_head_id = [d.head_id for d in batch_dict['batch_data']]
            total_tail_id = [d.tail_id for d in batch_dict['batch_data']]
            tail_vector = []
            with torch.no_grad():
                for temp_index in candidate_index:
                    indices_to_remove = []
                    temp_data = total_train_batch[temp_index].copy()
                    for f in range(len(temp_data['batch_data'])):
                        if temp_data['batch_data'][f].tail_id in total_tail_id:
                            indices_to_remove.append(f)
                        elif self.args.use_self_negative and temp_data['batch_data'][f].tail_id in total_head_id:
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
            if len(candidate_index) > 0:
                batch_dict['triplet_mask'] = construct_mask_extra_batch([ex for ex in batch_dict['batch_data']].copy(),
                                                                        total_tail_id.copy())
            if self.args.add_hop_mask > 0:
                temp_mask = construct_n_hop_mask(total_head_id, total_tail_id, n_hop=self.args.add_hop_mask)
                batch_dict['triplet_mask'] = batch_dict['triplet_mask'] & temp_mask
            if torch.cuda.is_available():
                tail_vector = move_to_cuda(tail_vector)
                batch_dict = move_to_cuda(batch_dict)
            if self.args.pretrained_ckpt:
                self.model.eval()
            else:
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
            # loss = self.criterion(logits, labels)
            loss1 = self.criterion(logits, labels)
            loss3 = self.criterion(logits[:, :batch_size].t(), labels)
            loss = loss1 + loss3
            self.optimizer.zero_grad()
            if not self.args.pretrained_ckpt:
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

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            inv_t.update(outputs.inv_t, 1)
            losses.update(loss.item(), batch_size)
            if self.args.add_discriminator:
                total_head=batch_dict['head_text']
                total_rel=batch_dict['rel_text']
                total_tail=batch_dict['tail_text']
                topk=2
                rand_n=2
                top_values, top_indices = torch.topk(logits, topk, dim=1)
                triples=[]
                total_labels=[]
                for piece in range(batch_size):
                    indices = torch.randperm(topk)[:rand_n]
                    index_array = top_indices[piece][indices]
                    index_array = torch.cat((index_array, torch.tensor([piece]).to(index_array.device)))
                    indices = torch.randperm(rand_n + 1)[:rand_n + 1]
                    index_array = index_array[indices]
                    for index in index_array:
                        if index==batch_size:
                            triple = ['the head is ' + total_head[piece] + ', the relation is ' + total_rel[
                                piece], 'the predict tail is ' + total_head[piece]]
                            triples.append(triple)
                        else:
                            triple=['the head is '+total_head[piece]+', the relation is '+total_rel[piece],'the predict tail is '+total_tail[index]]
                            triples.append(triple)
                        if piece != index:
                            total_labels.append(0)
                        else:
                            total_labels.append(1)
                mini_batch=300
                for index in range(0, len(triples), mini_batch):
                    if index+mini_batch > len(triples):
                        mini_batch=len(triples)-index
                    temp_triples=triples[index:index+mini_batch]
                    outputs = self.model.module.discriminate(temp_triples)
                    results=outputs
                    labels_dis = torch.tensor(total_labels[index:index + mini_batch])
                    labels_dis = move_to_cuda(labels_dis).to(torch.float)
                    loss = self.criterion2(results, labels_dis)
                    acc1_dis = new_accuracy(results, labels_dis)
                    top1_dis.update(acc1_dis,mini_batch)
                    losses_dis.update(loss.item(), mini_batch)
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
                # compute gradient and do SGD step
            if i % self.args.print_freq == 0:
                if self.args.add_discriminator:
                    progress_dis.display(i)
                else:
                    progress.display(i)
                if self.extra_flag:
                    if acc1 > 98:
                        logger.info("acc1已超过98%,添加额外待预测的尾实体")

                        if self.extra_batch_size == 0 and self.extra_batch_limit != 0:
                            self.extra_batch_size = 1
                            logger.info("尾实体添加成功,当前额外batch数量为" + str(self.extra_batch_size))
                            self.reset_learning_rate(self.train_steps)
                            return True
                        else:
                            if self.extra_batch_size < self.extra_batch_limit:
                                self.extra_batch_size *= 2
                                if self.extra_batch_size > self.extra_batch_limit:
                                    self.extra_batch_size = self.extra_batch_limit
                                logger.info("尾实体添加成功,当前额外batch数量为" + str(self.extra_batch_size))
                                self.reset_learning_rate(self.train_steps)
                                return True
                            else:
                                logger.info("尾实体数量已达到预定义上限,修改请参考extra-batch-limit参数")
                                self.extra_flag = False
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
        return False

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
    def predict_by_examples(self, examples: List[Example], entities_tensor):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.args.task),
            num_workers=1,
            batch_size=max(self.args.batch_size, 512),
            collate_fn=collate,
            shuffle=False)
        hr_tensor_list = []
        for idx, batch_dict in enumerate(data_loader):
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
        return torch.cat(hr_tensor_list, dim=0)

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
            batch_dict['return_direct'] = True
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)

    @torch.no_grad()
    def eval_single_direction(self,
                              entity_tensor: torch.tensor,
                              eval_forward=True,
                              batch_size=1024) -> dict:
        start_time = time()
        examples = load_data(self.args.valid_path, add_forward_triplet=eval_forward,
                             add_backward_triplet=not eval_forward)
        hr_tensor = self.predict_by_examples(examples, entity_tensor)
        # if not self.args.use_cross_attention:
        #     hr_tensor, _ = self.predict_by_examples(examples)
        # else:
        #     hr_tensor = self.predict_by_examples_new(all_entity_exs = entity_dict.entity_exs, valid_examples = examples)
        hr_tensor = hr_tensor.to(entity_tensor.device)
        target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
        logger.info('predict tensor done, compute metrics...')
        print(hr_tensor.shape, entity_tensor.shape)
        topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor,
                                                                    entities_tensor=entity_tensor,
                                                                    target=target, examples=examples,
                                                                    batch_size=batch_size,
                                                                    model=self.model.module)
        eval_dir = 'forward' if eval_forward else 'backward'
        logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))
        logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
        if self.args.add_discriminator:
            total_head=[]
            total_rel=[]
            for ex in examples:
                test_data=ex.vectorize()
                total_head.append(test_data['head_text'])
                total_rel.append(test_data['rel_text'])
            total_tail=[_concat_name_desc(ex.entity,ex.entity_desc)for ex in entity_dict.entity_exs]
            topk_scores, topk_indices, metrics, ranks = compute_metrics1(hr_tensor=hr_tensor,
                                                                        entities_tensor=entity_tensor,
                                                                        target=target, examples=examples,
                                                                        batch_size=batch_size,
                                                                        model=self.model.module,
                                                                         total_head=total_head,
                                                                         total_rel=total_rel,
                                                                         total_tail=total_tail
                                                                        )
            logger.info('使用判别器')
            logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))
            logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))

        return metrics
