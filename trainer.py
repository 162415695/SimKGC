import glob
import json
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.utils.data

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from copy import deepcopy

from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj, \
    concatenate_dict_arrays, generate_random_numbers
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()
        self.extra_batch_size = 0
        self.extra_flag = True
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
            print("额外batch上限因为数据量调整为" + str(self.extra_batch_limit))

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

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
        if len(total_tail_id) != len(set(total_tail_id)):
            print("xxx special error")
        tail_vector = []
        with torch.no_grad():
            for temp_index in candidate_index:
                indices_to_remove = []
                temp_data = total_valid_batch[temp_index]
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
        print('当前额外batch为' + str(self.extra_batch_size) + ',目前尾实体数量为' + str(len(total_tail_id)))
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
            model.tail_bert = deepcopy(model.hr_bert)
            candidate_index = generate_random_numbers(self.extra_batch_size, i, len(total_train_batch))
            self.model.eval()
            total_tail_id = [d.tail_id for d in batch_dict['batch_data']]
            print(len(total_tail_id))
            print(len(set(total_tail_id)))
            if len(total_tail_id)!=len(set(total_tail_id)):
                print("xxx special error")
            tail_vector = []
            with torch.no_grad():
                for temp_index in candidate_index:
                    indices_to_remove = []
                    temp_data = total_train_batch[temp_index]
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
            print('当前额外batch为' + str(self.extra_batch_size) + ',目前尾实体数量为' +str(len(total_tail_id)))
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
            if self.extra_flag:
                if acc1 > 90:
                    print("acc1已超过90%,添加额外待预测的尾实体")
                    if self.extra_batch_size==0:
                        self.extra_batch_size =1
                    else:
                        if self.extra_batch_size < self.extra_batch_limit:
                            self.extra_batch_size *= 2
                            if self.extra_batch_size > self.extra_batch_limit:
                                self.extra_batch_size = self.extra_batch_limit
                            print("尾实体添加成功,当前额外batch数量为" + str(self.extra_batch_size))
                        else:
                            print("尾实体数量已达到预定义上限,修改请参考extra-batch-limit参数")
                            self.extra_flag = False
            if i % self.args.print_freq == 0:
                progress.display(i)

            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)
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
