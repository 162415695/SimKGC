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

import utils
from hop_graph import graph_build
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
from dict_hub import build_tokenizer, all_triplet_dict
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


def cosine_similarity_loss(u, v):
    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(u, v, dim=-1)
    # 损失值：1 - 相似度
    loss = 1 - cosine_sim.mean()
    return loss


class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # 初始化教师模型的中心向量
        self.register_buffer("center", torch.zeros(1, 1, out_dim))  # 中心化向量

        # 教师温度调度表
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch=0):
        """
        student_output: batch × num_tokens × dim
        teacher_output: batch × num_tokens × dim
        """
        self.center = self.center.to(teacher_output.device)

        # 1. 学生模型输出的温度缩放
        student_out = student_output / self.student_temp  # 温度缩放
        # 对最后一个维度（特征维度 dim）进行 log_softmax
        student_out = F.log_softmax(student_out, dim=-1)

        # 2. 教师模型输出的温度缩放和 softmax
        temp = self.teacher_temp_schedule[epoch]  # 当前 epoch 的教师温度
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)  # softmax 归一化
        teacher_out = teacher_out.detach()  # 分离计算图，避免反向传播到教师模型

        # 3. 损失计算（token-wise）
        # 逐 token 计算交叉熵损失，-q * log(p)，最后对 batch 和 token 求平均
        loss = torch.sum(-teacher_out * student_out, dim=-1)  # 每个 token 的损失
        loss = loss.mean()  # 对 batch 和 num_tokens 求平均

        # 4. 更新教师模型的中心向量
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        更新教师模型输出的中心向量
        """
        # 在 DP 模式下，不需要分布式同步，仅对当前 GPU 的 batch 进行操作
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)  # 对 batch 维度求均值

        # 动态更新中心向量（使用动量平滑）
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def random_sample_with_replacement(tensor, num_samples):
    indices = torch.randint(0, tensor.size(0), (num_samples,), device=tensor.device)
    return tensor[indices]


class Trainer:

    def __init__(self, args, ngpus_per_node):
        graph_build()
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
        self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean').cuda()
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
        examples = train_dataset.examples
        self.train_keys = []
        self.train_examples = {}
        for i in examples:
            key = i.head_id + i.relation
            self.train_keys.append(key)
            if key in self.train_examples:
                self.train_examples[key].append(i)
            else:
                self.train_examples[key] = [i]
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
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)
        self.extra_batch_limit = args.extra_batch_limit
        if self.extra_batch_limit == -1 or self.extra_batch_limit > len(self.train_loader):
            self.extra_batch_limit = len(self.train_loader) - 1
            logger.info("额外batch上限因为数据量调整为" + str(self.extra_batch_limit))
        self.momentum_schedule_mlp = utils.cosine_scheduler(self.args.ema_decay_mlp, 1, args.dino_stop_epochs,
                                                            len(self.train_loader))
        self.momentum_schedule_bert = utils.cosine_scheduler(self.args.ema_decay_bert, 1, args.dino_stop_epochs,
                                                             len(self.train_loader))

    def compute_dino_loss_and_weights(self,
                                      student_hr, student_tail, teacher_tail,
                                      momentum=0.9, num_negatives=100, pos_weight=1.0, strong_neg_weight=3.0,
                                      weak_neg_weight=1.0, proportion=0.5,
                                      temperature=0.07):
        """
        计算两个学生模型与教师模型的 DINO 损失矩阵，基于损失选择负样本，生成权重矩阵，并更新 Center。

        参数：
        - student_hr: 学生模型的第一个输出，形状为 [batch_size, embedding_dim]
        - student_tail: 学生模型的第二个输出，形状为 [batch_size, embedding_dim]
        - teacher_tail: 教师模型的输出，形状为 [batch_size, embedding_dim]
        - center: 教师的中心向量，形状为 [embedding_dim]
        - momentum: Center 更新的动量系数
        - num_negatives: 需要选择的负样本总数
        - pos_weight: 正样本的权重
        - strong_neg_weight: 显著负样本的权重
        - weak_neg_weight: 其他负样本的权重
        - proportion: 从第一个损失矩阵中选择负样本的比例
        - temperature: 温度参数，用于调整教师 softmax 的分布平滑度

        返回：
        - loss_matrix1: 学生1与教师模型之间的损失矩阵
        - loss_matrix2: 学生2与教师模型之间的损失矩阵
        - weight_matrix1: 学生1的权重矩阵
        - weight_matrix2: 学生2的权重矩阵
        - updated_center: 更新后的中心向量
        """
        # 获取 batch_size
        batch_size = student_hr.size(0)
        ########## 1. 调整教师输出（减去 center） ##########
        teacher_tail=teacher_tail.detach()
        teacher_tail_centered = teacher_tail - self.model.module.center  # 对教师输出进行中心化

        ########## 2. 计算相似度矩阵 ##########
        # 学生与教师的相似度矩阵
        student_sim1 = F.cosine_similarity(student_hr[:, None, :], teacher_tail_centered[None, :, :], dim=-1)
        student_sim2 = F.cosine_similarity(student_tail[:, None, :], teacher_tail_centered[None, :, :], dim=-1)
        # 教师的相似度矩阵
        teacher_sim = F.cosine_similarity(teacher_tail_centered[:, None, :], teacher_tail_centered[None, :, :], dim=-1)

        ########## 3. 计算 softmax 和 log-softmax ##########
        # 计算学生模型的 log-softmax 概率分布
        student_log_prob1 = F.log_softmax(student_sim1, dim=-1)
        student_log_prob2 = F.log_softmax(student_sim2, dim=-1)
        # 计算教师模型的 softmax 概率分布（带温度调节）
        teacher_prob = F.softmax(teacher_sim / temperature, dim=-1)

        ########## 4. 计算 DINO 损失矩阵 ##########
        # 交叉熵公式：Loss = - sum(q * log(p))
        loss_matrix1 = - (teacher_prob * student_log_prob1) # 学生1与教师的损失矩阵
        loss_matrix2 = - (teacher_prob * student_log_prob2)  # 学生2与教师的损失矩阵

        ########## 5. 初始化权重矩阵 ##########
        # 初始化为弱负样本的权重
        weight_matrix1 = torch.full((batch_size, batch_size), weak_neg_weight, device=student_hr.device)
        weight_matrix2 = torch.full((batch_size, batch_size), weak_neg_weight, device=student_hr.device)
        # 设置正样本（对角线元素）的权重
        weight_matrix1[range(batch_size), range(batch_size)] = pos_weight
        weight_matrix2[range(batch_size), range(batch_size)] = pos_weight
        ########## 6. 选取负样本 ##########
        # 创建布尔掩码，排除对角线（正样本）
        mask = torch.eye(batch_size, dtype=torch.bool, device=student_hr.device)

        # 拉平矩阵并去掉正样本
        loss1_flat = loss_matrix1[~mask].view(batch_size, -1)
        loss2_flat = loss_matrix2[~mask].view(batch_size, -1)

        # 从第一个损失矩阵中选取显著负样本
        num_negatives1 = int(num_negatives * proportion)  # 从第一个矩阵中选择的负样本数量
        topk_indices1 = torch.topk(loss1_flat.view(-1), num_negatives1, largest=True).indices
        weight_matrix1.view(-1)[topk_indices1] = strong_neg_weight

        # 从第二个损失矩阵中选取显著负样本
        num_negatives2 = num_negatives - num_negatives1  # 剩余负样本数量
        topk_indices2 = torch.topk(loss2_flat.view(-1), num_negatives2, largest=True).indices
        weight_matrix2.view(-1)[topk_indices2] = strong_neg_weight

        ########## 7. 更新 Center ##########
        # 计算教师输出的均值
        teacher_mean = teacher_tail.mean(dim=0)
        # 更新 Center
        self.model.module.center = self.model.module.center * (1 - momentum) + teacher_mean * momentum

        ########## 8. 返回结果 ##########
        return loss_matrix1, loss_matrix2, weight_matrix1, weight_matrix2

    def compute_classwise_loss_matrix(self,logit, label):
        """
        计算每个样本的每个类别的交叉熵损失矩阵。

        参数:
            logit (torch.Tensor): 形状为 [batch, num_classes] 的预测分数张量。
            label (torch.Tensor): 形状为 [batch] 的真实类别索引张量。

        返回:
            torch.Tensor: 形状为 [batch, num_classes] 的损失矩阵。
        """
        # Step 1: 计算 log-softmax，形状为 [batch, num_classes]
        log_softmax = F.log_softmax(logit, dim=1)

        # Step 2: 创建 one-hot 标签矩阵，形状为 [batch, num_classes]
        # 对于每个样本，将目标类别索引转化为 one-hot 向量
        # Step 2: 构造对角选择矩阵，形状为 [batch, num_classes]
        batch_size, num_classes = logit.shape
        diag_matrix = torch.zeros_like(logit)  # 初始化为全 0 矩阵
        diag_matrix[torch.arange(batch_size), label] = 1  # 对角线位置设置为 1
        # Step 3: 根据交叉熵公式计算损失矩阵
        # 交叉熵公式： - label[i, k] * log_softmax[i, k]
        loss_matrix = -diag_matrix * log_softmax  # [batch, num_classes]

        return loss_matrix

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        epoch = 0

        # self._run_eval(epoch=0, extra_batch_num=0)
        eval_flag = 0
        while epoch < self.args.epochs:
            # train for one epoch
            extra_flag, loss = self.train_epoch(epoch)
            if extra_flag:
                logger.info('已扩大batch,重新进行训练')
                epoch = 0  # 重置为0重新开始
            else:
                epoch += 1  # 继续到下一个epoch
               # if epoch > self.args.epochs / 2 or epoch % 10 == 0 or eval_flag > 0.5:
                eval_flag = self._run_eval(epoch=epoch, extra_batch_num=self.extra_batch_size)

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
        return metric_dict['hit@1']

    @torch.no_grad()
    def eval_entity(self, epoch) -> Dict:
        self.model.eval()
        _convert_is_test_2_true()
        #entity_tensor = self.predict_by_entities(entity_dict.entity_exs)
        forward_metrics = self.eval_single_direction(eval_forward=True)
        backward_metrics = self.eval_single_direction(eval_forward=False)
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

    def loss_function(self,matrix, lambda_factor=10, beta_factor=1, delta_factor=1, C=1.0,print=False):
        diag_loss = torch.diagonal(matrix).mean()

        # 非对角线元素
        N = matrix.size(0)
        eye_mask = torch.eye(N, device=matrix.device)
        non_diag_elements = matrix * (1 - eye_mask)  # 只保留非对角线元素
        non_diag_loss = non_diag_elements.mean()

        # 计算最小值
        non_diag_values = non_diag_elements[non_diag_elements > 0]
        min_loss = non_diag_values.min()
        '''
        logger.info('对角线'+str(lambda_factor * diag_loss))
        logger.info('非对角线线' + str(beta_factor * (C - non_diag_loss)))
        logger.info('正则化' + str(delta_factor * min_loss))
        '''
        # 总损失
        total_loss = (lambda_factor * diag_loss -
                      beta_factor * non_diag_loss )
        return total_loss
    def loss_backward(self,loss,compute_graph=False):
        if self.args.use_amp:
            self.scaler.scale(loss).backward(retain_graph=compute_graph)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward(retain_graph=compute_graph)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
    def loss_compute(self,outputs,batch_dict,batch_size):
        model = get_model_obj(self.model)
        with torch.cuda.amp.autocast():


            outputs_new= model.compute_logits(output_dict=outputs, batch_dict=batch_dict)
        outputs_new = ModelOutput(**outputs_new)
        logits, labels = outputs_new.logits,outputs_new.labels
        assert logits.size(0) == batch_size
        # head + relation -> tail
        # loss = self.criterion(logits, labels)

        loss1 = self.criterion(logits, labels)
        loss3 = self.criterion(logits[:, :batch_size].t(), labels)
        loss = loss1 + loss3
        return loss
    def train_epoch(self, epoch):
        self.model.train()
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
        if self.args.use_dino:
            progress = ProgressMeter(
                len(self.train_loader),
                [losses, top1, top3],
                prefix=prefix)

        if self.args.add_discriminator:
            prefix = "Epoch: [{}],discriminator: ".format(epoch)
            losses_dis = AverageMeter('Loss', ':.4')
            top1_dis = AverageMeter('Acc@1', ':6.2f')
            progress_dis = ProgressMeter(
                len(self.train_loader),
                [losses_dis, top1_dis],
                prefix=prefix
            )
        if self.extra_batch_size > 0:
            total_train_batch = {i: k for i, k in enumerate(self.train_loader)}
        i = 0
        for batch_dict in self.train_loader:
            self.current_steps += 1
            model = get_model_obj(self.model)
            if self.extra_batch_size > 0:
                candidate_index = generate_random_numbers(self.extra_batch_size, -1, len(total_train_batch))

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
                    batch_dict['triplet_mask'] = construct_mask_extra_batch(
                        [ex for ex in batch_dict['batch_data']].copy(),
                        total_tail_id.copy())
                if self.args.add_hop_mask > 0:
                    temp_mask = construct_n_hop_mask(total_head_id, total_tail_id, n_hop=self.args.add_hop_mask)
                    batch_dict['triplet_mask'] = batch_dict['triplet_mask'] & temp_mask
                if torch.cuda.is_available():
                    tail_vector = move_to_cuda(tail_vector)
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            '''
            if self.args.pretrained_ckpt:
                self.model.eval()
            else:
                self.model.train()
                '''
            batch_size = len(batch_dict['batch_data'])
            # compute output

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)

            if not self.args.use_dino:

                with torch.cuda.amp.autocast():
                    if self.extra_batch_size > 0:
                        outputs_new = model.compute_logits(output_dict=outputs, batch_dict=batch_dict,
                                                       extra_tail=tail_vector)
                    else:
                        outputs_new = model.compute_logits(output_dict=outputs, batch_dict=batch_dict)
                outputs_new = ModelOutput(**outputs_new)
                logits, labels = outputs_new.logits, outputs_new.labels
                assert logits.size(0) == batch_size
                # head + relation -> tail
                # loss = self.criterion(logits, labels)

                loss1 = self.criterion(logits, labels)
                loss3 = self.criterion(logits[:, :batch_size].t(), labels)
                loss = loss1 + loss3
                acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
                top1.update(acc1.item(), batch_size)
                top3.update(acc3.item(), batch_size)
                inv_t.update(outputs_new.inv_t, 1)
                losses.update(loss.item(), batch_size)


            else:
                loss_matrix1, loss_matrix2, weight_matrix1, weight_matrix2 = self.compute_dino_loss_and_weights(
                    outputs['hr_vector'], outputs['tail_vector'], outputs['teacher_vector'],
                    num_negatives=self.args.hard_negative_num
                )
                diagonal_length = min(weight_matrix1.size(0), weight_matrix1.size(1))
                diag_indices = torch.arange(diagonal_length)  # 对角线索引
                factor = 1
                weight_matrix1[diag_indices, diag_indices] *= factor
                weight_matrix2[diag_indices, diag_indices] *= factor


                with torch.cuda.amp.autocast():
                        outputs_new = model.compute_logits(output_dict=outputs, batch_dict=batch_dict,
                                                       )
                outputs_new = ModelOutput(**outputs_new)
                logits, labels = outputs_new.logits, outputs_new.labels



                 # 使用 one-hot 矩阵选择对应的类别概率，并计算交叉熵

                weight = torch.maximum(weight_matrix1, weight_matrix2)
                if self.args.use_self_negative:
                        temp_self=torch.full((batch_size,), weight.min().item()).unsqueeze(1)
                        temp_self = temp_self.to(weight.device)
                        new_weight=torch.cat([weight,temp_self ],dim=-1)

                if self.args.dino_loss:
                    loss_matrix1*=weight_matrix1
                    loss_matrix2*=weight_matrix2
                dino_loss=self.loss_function(loss_matrix1)+self.loss_function(loss_matrix2)

                loss1 = self.compute_classwise_loss_matrix(logits, labels)
                loss3 = self.compute_classwise_loss_matrix(logits[:, :batch_size].t(), labels)

                if self.args.use_self_negative:
                    loss1*=new_weight
                else:
                    loss1*=weight
                loss3*=weight
                contrastive_loss=torch.sum(loss1,dim=-1).mean()+torch.sum(loss3,dim=-1).mean()
                if self.args.dino_loss:
                    contrastive_loss/=1e10
                    loss=contrastive_loss+dino_loss
                else:
                    if epoch < self.args.dino_epochs:
                            loss = contrastive_loss + dino_loss
                    elif epoch < self.args.dino_stop_epochs:
                            loss = contrastive_loss + dino_loss * ((epoch - self.args.dino_epochs) / (
                                        self.args.dino_stop_epochs - self.args.dino_epochs))
                    else:
                            loss = contrastive_loss
                inv_t.update(outputs_new.inv_t, 1)


                acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
                top1.update(acc1.item(), batch_size)
                top3.update(acc3.item(), batch_size)

                '''
                sample1 = outputs['hr_vector'].clone()
                sample2 = outputs['tail_vector'].clone()
                sample1= random_sample_with_replacement(sample1,len(sample1))
                sample2 = random_sample_with_replacement(sample2,len(sample2))
                nega_loss=self.dino_loss(sample1, sample2)
                loss = self.dino_loss(outputs['hr_vector'],outputs['tail_vector'])

                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                    nega_loss = nega_loss.mean()

                loss += (1 - nega_loss)
                losses.update(loss.item(), batch_size)
                '''
            self.optimizer.zero_grad()
            #if not self.args.pretrained_ckpt:
            if not self.args.hr_negative:
                self.loss_backward(loss)
            else:
                #self.loss_backward(loss,True)
                hr_output = {}
                hr_output['tail_vector'] = outputs['hr_vector'].clone()  # 确保克隆
                hr_output['hr_vector'] = outputs['hr_vector'].clone()  # 确保克隆
                hr_output['head_vector'] = outputs['head_vector'].clone()
                hr_loss=self.loss_compute(hr_output,batch_dict,batch_size)
                #self.loss_backward(hr_loss,True)
                tail_output = {}
                tail_output['tail_vector'] = outputs['tail_vector'].clone()  # 确保克隆
                tail_output['hr_vector'] = outputs['tail_vector'].clone()  # 确保克隆
                tail_output['head_vector'] = outputs['head_vector'].clone()
                tail_loss=self.loss_compute(tail_output,batch_dict,batch_size)
                loss=loss+hr_loss+tail_loss
                self.loss_backward(loss, False)
            losses.update(loss.item(), batch_size)

            if self.args.add_discriminator:
                total_head = batch_dict['head_text']
                total_rel = batch_dict['rel_text']
                total_tail = batch_dict['tail_text']
                topk = 2
                rand_n = 2
                top_values, top_indices = torch.topk(logits, topk, dim=1)
                triples = []
                total_labels = []
                for piece in range(batch_size):
                    indices = torch.randperm(topk)[:rand_n]
                    index_array = top_indices[piece][indices]
                    index_array = torch.cat((index_array, torch.tensor([piece]).to(index_array.device)))
                    indices = torch.randperm(rand_n + 1)[:rand_n + 1]
                    index_array = index_array[indices]
                    for index in index_array:
                        if index == batch_size:
                            triple = ['the head is ' + total_head[piece] + ', the relation is ' + total_rel[
                                piece], 'the predict tail is ' + total_head[piece]]
                            triples.append(triple)
                        else:
                            triple = ['the head is ' + total_head[piece] + ', the relation is ' + total_rel[piece],
                                      'the predict tail is ' + total_tail[index]]
                            triples.append(triple)
                        if piece != index:
                            total_labels.append(0)
                        else:
                            total_labels.append(1)
                mini_batch = 300
                for index in range(0, len(triples), mini_batch):
                    if index + mini_batch > len(triples):
                        mini_batch = len(triples) - index
                    temp_triples = triples[index:index + mini_batch]
                    outputs = self.model.module.discriminate(temp_triples)
                    results = outputs
                    labels_dis = torch.tensor(total_labels[index:index + mini_batch])
                    labels_dis = move_to_cuda(labels_dis).to(torch.float)
                    loss = self.criterion2(results, labels_dis)
                    acc1_dis = new_accuracy(results, labels_dis)
                    top1_dis.update(acc1_dis, mini_batch)
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
            i += 1
        if self.args.use_dino and epoch < self.args.dino_stop_epochs:
            #if self.current_steps %100==0:
            if epoch < self.args.dino_warmup_epochs and not self.args.dino_loss:
                m_mlp = 0
                m_bert = 0
            else:
                m_mlp = self.momentum_schedule_mlp[self.current_steps - 1]  # 当前动量值
                m_bert = self.momentum_schedule_bert[self.current_steps - 1]  # 当前动量值
            with torch.no_grad():
                '''
                    if self.args.pretrained_ckpt:
                        for param_q, param_k in zip(self.model.module.hr_bert.dino_head.parameters(),
                                                    self.model.module.teacher_model.dino_head.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                    else:
'''
                m = m_mlp
                for param_hr, param_tail, param_teacher in zip(self.model.module.hr_bert.dino_head.parameters(),
                                                               self.model.module.tail_bert.dino_head.parameters(),
                                                               self.model.module.teacher_model.dino_head.parameters()):
                    param_teacher.data.mul_(m).add_((1 - m) * (0.5 * param_hr.detach().data
                                                               + 0.5 * param_tail.detach().data))
                m = m_bert
                for param_hr, param_tail, param_teacher in zip(self.model.module.hr_bert.bert.parameters(),
                                                               self.model.module.tail_bert.bert.parameters(),
                                                               self.model.module.teacher_model.bert.parameters()):
                    param_teacher.data.mul_(m).add_((1 - m) * (0.5 * param_hr.detach().data
                                                               + 0.5 * param_tail.detach().data))


        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
        return False, loss.detach()

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
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)

    @torch.no_grad()
    def eval_single_direction(self,
                              entity_tensor: torch.tensor = None,
                              eval_forward=True,
                              batch_size=4096) -> dict:
        start_time = time()
        examples = load_data(self.args.valid_path, add_forward_triplet=eval_forward,
                             add_backward_triplet=not eval_forward)
        hr_tensor = self.predict_by_examples(examples, entity_tensor)
        # if not self.args.use_cross_attention:
        #     hr_tensor, _ = self.predict_by_examples(examples)
        # else:
        #     hr_tensor = self.predict_by_examples_new(all_entity_exs = entity_dict.entity_exs, valid_examples = examples)
        target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
        logger.info('predict tensor done, compute metrics...')
        k = 3
        temp_examples = []
        for entity_ex in entity_dict.entity_exs:
            temp_examples.append(Example(head_id='', relation='',
                                         tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=temp_examples, task=self.args.task),
            num_workers=2,
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=False)
        total = hr_tensor.size(0)
        entity_cnt = len(entity_dict)
        target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
        '''
        if self.args.use_dino:
            hr_tensor  = hr_tensor  / 0.1  # 温度缩放
            # 对最后一个维度（特征维度 dim）进行 log_softmax
            if self.args.dino_loss:
                hr_tensor  = F.log_softmax(hr_tensor , dim=-1)
            else:
                hr_tensor = F.log_softmax(hr_tensor , dim=-1)
'''
        start = 0
        all_scores = []
        # 以 tail 为基础进行批次处理
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            end = start + batch_size
            batch_dict['only_ent_embedding'] = True
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            entities_tensor = outputs['ent_vectors']
            if self.args.use_dino:
                entities_tensor = entities_tensor
            if self.args.use_cross_attention:
                batch_score = self.model.module.compute_score(hr_tensor, entities_tensor)
            else:
                batch_score = torch.mm(hr_tensor, entities_tensor.t())
            all_scores.append(batch_score)
        all_scores = torch.cat(all_scores, dim=1)  # total * entity_cnt
        for idx in range(all_scores.size(0)):
            mask_indices = []
            cur_ex = examples[idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_idx = entity_dict.entity_to_idx(e_id)
                mask_indices.append(mask_idx)
            if mask_indices:
                mask_indices = torch.LongTensor(mask_indices).to(all_scores.device)
                all_scores[idx].index_fill_(0, mask_indices, -1)

        sorted_scores, sorted_indices = torch.sort(all_scores, dim=-1, descending=True)  # total * entity_cnt
        target_rank = torch.nonzero(sorted_indices.eq(target), as_tuple=False)
        # 初始化统计指标
        mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
        ranks = []
        topk_scores, topk_indices = [], []
        assert target_rank.size(0) == all_scores.size(0)
        for idx in range(all_scores.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]
            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)
            topk_scores.append(sorted_scores[idx, :k].tolist())
            topk_indices.append(sorted_indices[idx, :k].tolist())
        # 计算最终指标
        metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
        metrics = {k: round(v / total, 4) for k, v in metrics.items()}
        assert len(topk_scores) == total

        eval_dir = 'forward' if eval_forward else 'backward'
        logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))
        logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
        if self.args.add_discriminator:
            total_head = []
            total_rel = []
            for ex in examples:
                test_data = ex.vectorize()
                total_head.append(test_data['head_text'])
                total_rel.append(test_data['rel_text'])
            total_tail = [_concat_name_desc(ex.entity, ex.entity_desc) for ex in entity_dict.entity_exs]
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
