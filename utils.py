import os
import glob
import torch
import shutil

import numpy as np
import torch.nn as nn
import random
from logger_config import logger



class AttrDict:
    pass


def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)

def copy_checkpoint(filename: str, is_best: bool):
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.mdl')
    shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.mdl')
def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint {}'.format(f))
        os.system('rm -f {}'.format(f))


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_parameters += np.prod(list(p.size()))
            logger.info('{}: {}'.format(name, np.prod(list(p.size()))))

    logger.info('Number of parameters: {}M'.format(num_parameters // 10 ** 6))
    return num_parameters


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence

def pad_and_stack(sequences, padding_value=0):
    # Find the maximum length
    max_length = max(seq.size(1) for seq in sequences)

    # Pad each sequence to the maximum length
    padded = [torch.cat([seq, torch.full((seq.size(0), max_length - seq.size(1)), padding_value)], dim=1)
              if seq.size(1) < max_length else seq for seq in sequences]

    # Stack them into a single tensor
    return torch.stack(padded)
def concatenate_dict_arrays(dict_list, another_dict):
    result = {}

    for key in another_dict:
        if key in ['tail_token_ids', 'tail_mask', 'tail_token_type_ids']:
            arrays_to_concatenate = []

            # 添加 another_dict 中的数据
            arrays_to_concatenate.append(another_dict[key])

            for d in dict_list:
                if key in d:
                    arrays_to_concatenate.append(d[key])
            shapes = [array.shape for array in arrays_to_concatenate]
            logger.info(f"Shapes for key '{key}' : {shapes}")

            # 使用 pad_and_stack 函数进行填充
            padded_arrays = pad_and_stack(arrays_to_concatenate)
            logger.info('+++++++++')
            # 检查填充后的形状是否一致
            shapes = [array.shape for array in padded_arrays]
            logger.info(f"Shapes for key '{key}' after padding: {shapes}")

            if len(set(shape[1:] for shape in shapes)) != 1:
                raise ValueError(f"Arrays for key '{key}' have incompatible shapes after padding: {shapes}")
            logger.info('////////////////')
            # 合并数组
            combined_array = torch.cat([tensor for tensor in padded_arrays], dim=0)
            shapes = combined_array.shape
            logger.info(f"Shapes for key '{key}' after padding: {shapes}")

            result[key] = combined_array

    return result


def generate_random_numbers(count, not_equal, less_than):
    """
    生成指定数量的不重复且不等于指定值的小于给定值的非负整数。

    :param count: 生成数的数量。
    :param not_equal: 生成数不等于的值。
    :param less_than: 生成数小于的值。
    :return: 包含生成数的列表。
    """
    numbers = set()
    while len(numbers) < count:
        num = random.randint(0, less_than - 1)
        if num != not_equal:
            numbers.add(num)
    return list(numbers)
