from abc import ABC
from copy import deepcopy
from config import args
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from logger_config import logger
from triplet_mask import construct_mask
from utils import move_to_cuda


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.x2 = None
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, tail_mask=None):
        batch_size, in_dim1 = x1.size()
        # 计算 q1
        q1 = self.proj_q1(x1).view(batch_size, self.num_heads, self.k_dim).permute(1, 0,
                                                                                   2)  # num_head, batch_size_A, dim

        q1 = nn.functional.normalize(q1, dim=1)  # 计算 num_batches
        # 计算 k2
        k2 = self.proj_k2(x2).view(x2.size(0), self.num_heads, self.k_dim).permute(1, 2,
                                                                                   0)  # num_head, dim, batch_size_B
        k2 = nn.functional.normalize(k2, dim=1)
        v2 = self.proj_v2(x2).view(x2.size(0), self.num_heads, self.v_dim).permute(1, 0,
                                                                                   2)  # num_head, batch_size_B, dim
        v2 = nn.functional.normalize(v2, dim=1)
        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5  # num_head, batch_size_A, batch_size_B
        attn = nn.functional.normalize(attn, dim=1)
        output = torch.matmul(attn, v2).permute(1, 0, 2).contiguous().view(x1.size(0),
                                                                           -1)  # num_head, batch_size_A, dim -> batch_size_A, num_head, dim -> batch_size_A, num_head*dim
        output = nn.functional.normalize(self.proj_o(output), dim=1)
        # output = output.to(torch.float32)
        return output

    def eval_forward(self, x1, x2, tail_mask=None):
        batch_size, in_dim1 = x1.size()

        # 计算 q1
        q1 = self.proj_q1(x1).view(batch_size, self.num_heads, self.k_dim).permute(1, 0, 2)

        # 计算 num_batches
        num_batches = (x2.shape[0] + batch_size - 1) // batch_size
        total_k2 = []
        total_v2 = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, x2.shape[0])  # 确保不越界

            # 处理当前 batch 的 x2
            x2_current = x2[start_idx:end_idx]
            # 计算 k2
            k2_temp = self.proj_k2(x2_current).view(x2_current.size(0), self.num_heads, self.k_dim)
            v2_temp = self.proj_v2(x2_current).view(x2_current.size(0), self.num_heads, self.v_dim)
            total_v2.append(v2_temp)
            total_k2.append(k2_temp)
        k2 = torch.cat(total_k2, dim=0).permute(1, 2, 0)
        v2 = torch.cat(total_v2, dim=0).permute(1, 0, 2)
        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(1, 0, 2).contiguous().view(x1.size(0), -1)
        output = self.proj_o(output)
        # output = output.to(torch.float32)
        return output


class CrossAttentionSimple(nn.Module):
    def __init__(self, in_dim1, in_dim2, dim):
        super(CrossAttentionSimple, self).__init__()
        self.dim = dim
        self.proj_q1 = nn.Linear(in_dim1, dim, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, dim, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, dim, bias=False)
        self.proj_o = nn.Linear(dim, in_dim1)

    def forward(self, x1, x2):
        batch_size, in_dim1 = x1.size()
        # 计算 q1
        q1 = nn.functional.normalize(self.proj_q1(x1), dim=1)  # batch_size_A, dim
        # 计算 num_batches
        # 计算 k2
        k2 = nn.functional.normalize(self.proj_k2(x2).permute(1, 0), dim=1)  # dim, batch_size_B
        # 计算 v2
        v2 = nn.functional.normalize(self.proj_v2(x2), dim=1)  # batch_size_B, dim
        attn = torch.matmul(q1, k2)  # batch_size_A, batch_size_B
        attn = nn.functional.normalize(attn, dim=1)
        output = torch.matmul(attn, v2)  # batch_size_A, dim
        output = nn.functional.normalize(self.proj_o(output), dim=1)

        return output

    def test(self, x1, x2):
        q1 = nn.functional.normalize(self.proj_q1(x1), dim=1)  # batch_size_A, dim
        # 计算 num_batches
        # 计算 k2
        k2 = nn.functional.normalize(self.proj_k2(x2).permute(1, 0), dim=1)  # dim, batch_size_B
        # 计算 v2
        v2 = self.proj_v2(x2)  # batch_size_B, dim
        attn = torch.matmul(q1, k2)  # batch_size_A, batch_size_B
        return attn


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(self.args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = self.args.additive_margin
        self.batch_size = self.args.batch_size
        self.pre_batch = self.args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(self.args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
        self.cross_attention = CrossAttention(768, 768, 768, 768, 4)
        self.total_tail = None
        self.total_tail_mask = None
        if self.args.pretrained_ckpt:
            for param in self.tail_bert.parameters():
                param.requires_grad = False
            for param in self.hr_bert.parameters():
                param.requires_grad = False

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def _encode_without_pool(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                rel_token_ids, rel_mask, rel_token_type_ids,
                only_ent_embedding=False, return_direct=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        # if not self.args.use_cross_attention:
        #     hr_vector = self._encode(self.hr_bert,
        #                              token_ids=hr_token_ids,
        #                              mask=hr_mask,
        #                              token_type_ids=hr_token_type_ids)
        # else:
        #     # logger.info('rel_vector dtype: {}, head_vector dtype: {}'.format(rel_vector.dtype, head_vector.dtype))
        #     temp_hr = self._encode(self.hr_bert,
        #                            token_ids=hr_token_ids,
        #                            mask=hr_mask,
        #                            token_type_ids=hr_token_type_ids)
        #     temp_tail = self._encode(self.tail_bert,
        #                              token_ids=tail_token_ids,
        #                              mask=tail_mask,
        #                              token_type_ids=tail_token_type_ids)
        #     indices = torch.randperm(temp_tail.size(0))  # size(0) 是 1024
        #     # 使用生成的随机索引打乱张量
        #     temp_tail = temp_tail[indices]
        #     hr_vector = self.cross_attention(temp_hr, temp_tail)

        return {
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
        }

    def forward_cross_attention(self, hr_vector, tail_vector, head_vector=None) -> dict:
        # print(hr_vector.shape, tail_vector.shape, head_vector.shape)
        assert hr_vector.size(0) == self.batch_size
        indices = torch.randperm(tail_vector.size(0))  # size(0) 是 batch_size
        temp_tail = tail_vector[indices]
        temp_tail = temp_tail.detach()
        hr_vector = self.cross_attention(hr_vector, temp_tail)
        return {
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
        }

    def eval_forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                     tail_token_ids, tail_mask, tail_token_type_ids,
                     head_token_ids, head_mask, head_token_type_ids,
                     rel_token_ids, rel_mask, rel_token_type_ids,
                     only_ent_embedding=False, **kwargs) -> dict:
        # tail_vector = self._encode(self.tail_bert,
        #                            token_ids=tail_token_ids,
        #                            mask=tail_mask,
        #                            token_type_ids=tail_token_type_ids)
        # head_vector = self._encode(self.tail_bert,
        #                            token_ids=head_token_ids,
        #                            mask=head_mask,
        #                            token_type_ids=head_token_type_ids)

        # logger.info('rel_vector dtype: {}, head_vector dtype: {}'.format(rel_vector.dtype, head_vector.dtype))
        temp_hr = self._encode(self.hr_bert,
                               token_ids=hr_token_ids,
                               mask=hr_mask,
                               token_type_ids=hr_token_type_ids)
        # temp_tail = self._encode_without_pool(self.tail_bert,
        #                        token_ids=tail_token_ids,
        #                        mask=tail_mask,
        #                        token_type_ids=tail_token_type_ids)
        # indices = torch.randperm(temp_tail.size(0))  # size(0) 是 1024
        # # 使用生成的随机索引打乱张量
        # temp_tail = temp_tail[indices]
        self.total_tail = self.total_tail.to(temp_hr.device)
        hr_vector = self.cross_attention.eval_forward(temp_hr, self.total_tail)
        self.total_tail = self.total_tail.cpu()
        return {
            'hr_vector': hr_vector,
            # 'tail_vector': tail_vector,
            # 'head_vector': head_vector,
        }

    def compute_logits(self, output_dict: dict, batch_dict: dict, extra_tail=None) -> dict:
        if len(extra_tail) == 0:
            hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        else:
            total_tail = [output_dict['tail_vector']]
            for vector in extra_tail:
                total_tail.append(vector)
            hr_vector, tail_vector = output_dict['hr_vector'], torch.cat(total_tail, dim=0)

        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        logits = hr_vector.mm(tail_vector.t())
        if self.args.test_opinion:
            logits = self.cross_attention.test(hr_vector, tail_vector)

        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()
        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)
        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)
        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = (hr_vector.mm(self.pre_batch_vectors.clone().t()))
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
