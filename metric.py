import torch
import numpy as np
from typing import List


def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[torch.tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




def new_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算指定格式的 logits 和 labels 的准确率。

    参数:
        logits (torch.Tensor): 形状为 (m, n) 的张量，包含每个类别的预测分数。
        labels (torch.Tensor): 形状为 (m, n) 的张量，只包含 0 和 1，每行只有1个1。

    返回:
        float: 正确分类的比例。
    """
    with torch.no_grad():
        # 获取每行中最大 logit 的索引
        logits=torch.round(torch.sigmoid(logits))
        identical_elements = logits == labels

        # 计算相同元素的数量
        num_identical = identical_elements.sum().item()
        # 计算总元素数量
        total_elements = logits.numel()
        # 计算相同元素的比例
        proportion_identical = num_identical / total_elements

        return 100*proportion_identical
