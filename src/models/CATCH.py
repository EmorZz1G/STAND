import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

import torch
import torch.nn as nn


from typing import Tuple, Union

import numpy as np
import pandas as pd


def split_before(
    data: Union[pd.DataFrame, np.ndarray], index: int
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
                 Can be a pandas DataFrame or a NumPy array.
    :param index: Split index position.
    :return: Tuple containing the first and second parts of the data.
    """
    if isinstance(data, pd.DataFrame):
        return data.iloc[:index, :], data.iloc[index:, :]
    elif isinstance(data, np.ndarray):
        return data[:index, :], data[index:, :]
    else:
        raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")

def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        train_data_rest, valid_data = split_before(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        return train_data_value, valid_data_rest


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        elif mode == 'transform':
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

'''
* @author: EmpyreanMoon
*
* @create: 2024-08-25 20:20
*
* @description: various forms of frequency loss
'''

import torch
from einops import rearrange
import numpy as np
from torch.utils.data import DataLoader

class SegLoader(object):
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.test_labels = data

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.data[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def anomaly_detection_data_provider(data, batch_size, win_size=100, step=100, mode='train'):
    dataset = SegLoader(data, win_size, 1, mode)

    shuffle = False
    if mode == 'train' or mode == 'val':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False)
    return data_loader


class frequency_loss(torch.nn.Module):
    def __init__(self, configs, keep_dim=False, dim=None):
        super(frequency_loss, self).__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        if configs.auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif configs.auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError
        self.configs = configs
        if configs.mask:
            self._generate_mask()
        else:
            self.mask = None

    def _generate_mask(self):
        if self.configs.add_noise and self.configs.noise_amp > 0:
            seq_len = self.configs.pred_len
            cutoff_freq_percentage = self.configs.noise_freq_percentage
            if self.configs.auxi_mode == "rfft":
                cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            elif self.configs.auxi_mode == "fft":
                cutoff_freq = int((seq_len) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1)
        else:
            self.mask = None

    def forward(self, outputs, batch_y):
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)
        # fft shape: [B, P, D]
        if self.configs.auxi_type == 'complex':
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)
        elif self.configs.auxi_type == 'complex-phase':
            loss_auxi = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
        elif self.configs.auxi_type == 'complex-mag-phase':
            loss_auxi_mag = (frequency_outputs - self.fft(batch_y, dim=1)).abs()
            loss_auxi_phase = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        elif self.configs.auxi_type == 'phase':
            loss_auxi = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
        elif self.configs.auxi_type == 'mag':
            loss_auxi = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
        elif self.configs.auxi_type == 'mag-phase':
            loss_auxi_mag = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
            loss_auxi_phase = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise NotImplementedError

        if self.mask is not None:
            loss_auxi *= self.mask

        if self.configs.auxi_loss == "MAE":
            loss_auxi = loss_auxi.abs().mean(dim=self.dim,
                                             keepdim=self.keep_dim) if self.configs.module_first else loss_auxi.mean(
                dim=self.dim, keepdim=self.keep_dim).abs()  # check the dim of fft
        elif self.configs.auxi_loss == "MSE":
            loss_auxi = (loss_auxi.abs() ** 2).mean(dim=self.dim,
                                                    keepdim=self.keep_dim) if self.configs.module_first else (
                    loss_auxi ** 2).mean(dim=self.dim, keepdim=self.keep_dim).abs()
        else:
            raise NotImplementedError
        return loss_auxi


class frequency_criterion(torch.nn.Module):
    def __init__(self, configs):
        super(frequency_criterion, self).__init__()
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.seq_len
        self.patch_num = int((self.win_size - self.patch_size) / self.patch_stride + 1)
        self.padding_length = self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, outputs, batch_y):

        output_patch = outputs.unfold(dimension=1, size=self.patch_size,
                                      step=self.patch_stride)

        b, n, c, p = output_patch.shape
        output_patch = rearrange(output_patch, 'b n c p -> (b n) p c')
        y_patch = batch_y.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        y_patch = rearrange(y_patch, 'b n c p -> (b n) p c')

        main_part_loss = self.metric(output_patch, y_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, '(b n) p c -> b n p c', b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = torch.tensor([range(start_indices[i], end_indices[i]) for i in range(n)]).unsqueeze(0).unsqueeze(-1)
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros((b, n, self.win_size - self.padding_length, c)).to(main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / non_zero_cnt

        if self.padding_length > 0:
            padding_loss = self.metric(outputs[:, -self.padding_length:, :], batch_y[:, -self.padding_length:, :])
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss

'''
* @author: EmpyreanMoon
*
* @create: 2024-08-26 10:28
*
* @description: The structure of CATCH
'''

class SegLoader(object):
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.test_labels = data

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.data[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def anomaly_detection_data_provider(data, batch_size, win_size=100, step=100, mode='train'):
    dataset = SegLoader(data, win_size, 1, mode)

    shuffle = False
    if mode == 'train' or mode == 'val':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False)
    return data_loader

from torch import nn, einsum
from einops import rearrange
import math, torch
'''
* @author: EmpyreanMoon
*
* @create: 2024-09-02 17:29
*
* @description: the implementation of the dynamical contrastive loss
'''

import torch


class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def _stable_scores(self, scores):
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        stable_scores = scores - max_scores
        return stable_scores

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        cosine = (scores / norm_matrix).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask

        all_scores = torch.exp(cosine / self.temperature)

        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)),
                                                                p=1, dim=-1)
        loss = clustering_loss.mean(1) + self.k * regular_loss

        mean_loss = loss.mean()
        return mean_loss


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dynamical_contrastive_loss = None

        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)
        if attn_mask is not None:
            def _mask(scores, attn_mask):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask == 0, large_negative, 0)
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores

            masked_scores = _mask(scores, attn_mask)

            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        attn = self.attend(masked_scores * scale)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn, dynamical_contrastive_loss


class c_Transformer(nn.Module):  ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, regular_lambda=regular_lambda,
                                    temperature=temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        total_loss = 0
        for attn, ff in self.layers:
            x_n, attn, dcloss = attn(x, attn_mask=attn_mask)
            total_loss += dcloss
            x = x_n + x
            x = ff(x) + x
        dcloss = total_loss / len(self.layers)
        return x, attn, dcloss


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, regular_lambda=regular_lambda,
                                         temperature=temperature)

        self.mlp_head = nn.Linear(dim, d_model)  # horizon)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss  # ,attn
# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
* @author: EmpyreanMoon
*
* @create: 2024-09-02 17:32
*
* @description: 
'''
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax


class channel_mask_generator(torch.nn.Module):
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        self.generator = nn.Sequential(torch.nn.Linear(input_size * 2, n_vars, bias=False), nn.Sigmoid())
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):  # x: [(bs x patch_num) x n_vars x patch_size]

        distribution_matrix = self.generator(x)

        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)

        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag

        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)

        return resample_matrix


class CATCHModel(nn.Module):
    def __init__(self, configs,
                 **kwargs):
        super(CATCHModel, self).__init__()

        self.revin_layer = RevIN(configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last)
        # Patching
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = self.seq_len
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)
        # print("depth=",cf_depth)
        # Backbone
        self.re_attn = True
        self.mask_generator = channel_mask_generator(input_size=configs.patch_size, n_vars=configs.c_in)
        self.frequency_transformer = Trans_C(dim=configs.cf_dim, depth=configs.e_layers, heads=configs.n_heads,
                                       mlp_dim=configs.d_ff,
                                       dim_head=configs.head_dim, dropout=configs.dropout,
                                       patch_dim=configs.patch_size * 2,
                                       horizon=self.horizon * 2, d_model=configs.d_model * 2,
                                       regular_lambda=configs.regular_lambda, temperature=configs.temperature)

        # Head
        self.head_nf_f = configs.d_model * 2 * patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
                                    head_dropout=configs.head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
                                    head_dropout=configs.head_dropout)

        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)

        # break up R&I:
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def forward(self, z):  # z: [bs x seq_len x n_vars]
        z = self.revin_layer(z, 'norm')

        z = z.permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_size,
                       step=self.patch_stride)  # z1: [bs x nvars x patch_num x patch_size]
        z2 = z2.unfold(dimension=-1, size=self.patch_size,
                       step=self.patch_stride)  # z2: [bs x nvars x patch_num x patch_size]

        # for channel-wise_1
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        # model shape
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_size = z1.shape[3]

        # proposed
        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))  # z: [bs * patch_num,nvars, patch_size]
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))  # z: [bs * patch_num,nvars, patch_size]
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)

        z, dcloss = self.frequency_transformer(z_cat, channel_mask)
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)  # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)  # z: [bs x nvars x seq_len]
        z2 = self.head_f2(z2)  # z: [bs x nvars x seq_len]

        complex_z = torch.complex(z1, z2)

        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # denorm
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')

        return z, complex_z.permute(0, 2, 1), dcloss


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)  # z: [bs x seq_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x seq_len]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)

        return x

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import copy

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 20 else args.learning_rate * (0.5 ** ((epoch // 20) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * (0.5 ** ((epoch // 10) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {20: args.learning_rate * 0.5 , 40: args.learning_rate * 0.01, 60:args.learning_rate * 0.01,8:args.learning_rate * 0.01,100:args.learning_rate * 0.01 }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# class StandardScaler():
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def transform(self, data):
#         return (data - self.mean) / self.std

#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

# def test_params_flop(model,x_shape):
#     """
#     If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
#     """
#     model_params = 0
#     for parameter in model.parameters():
#         model_params += parameter.numel()
#         print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
#     from ptflops import get_model_complexity_info
#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
#         # print('Flops:' + flops)
#         # print('Params:' + params)
#         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#         print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "lr": 0.0001,
    "Mlr": 0.00001,
    "e_layers": 3,
    "n_heads": 2,
    "cf_dim": 64,
    "d_ff": 256,
    "d_model": 128,
    "head_dim": 64,
    "individual": 0,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "auxi_lambda": 0.005,
    "score_lambda": 0.05,
    "regular_lambda": 0.5,
    "temperature": 0.07,
    "patch_stride": 8,
    "patch_size": 16,
    "inference_patch_stride": 1,
    "inference_patch_size": 32,
    "dc_lambda": 0.005,
    "module_first": True,
    "mask": False,
    "pretrained_model": None,
    "num_epochs": 3,
    "batch_size": 128,
    "patience": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "seq_len": 192,
    "pct_start": 0.3,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "lradj": "type1",
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr

from .base import BaseDetector

class CATCH(BaseDetector):
    def __init__(self, **kwargs):
        super(CATCH, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.win_size = self.config.seq_len
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len
        self.if_save = kwargs.get("if_save", False)
        if self.if_save:
            print("CATCH will save the trained model.")
        else:
            print("CATCH will not save the trained model.")

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.to(device)

                output, _, _ = self.model(input)

                output = output[:, :, :]

                output = output.detach().cpu()
                true = input.detach().cpu()

                loss = criterion(output, true).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss
    
    def fit(self, X):
        X = pd.DataFrame(X)
        self.detect_fit(X, X)

        if self.if_save:
            # 直接保存这个实例
            torch.save(self, self.model_saving_path / f'CATCH_c{X.shape[1]}.pt')
    
    def decision_function(self, X) -> np.ndarray:
        X = pd.DataFrame(X)
        scores = self.detect_score(X)[0]
        if scores.shape[0] < len(X):
            reslen = len(X) - scores.shape[0]
            scores = np.concatenate((scores,np.ones(reslen)*scores[-1]),axis=0)

        return scores


    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train the model.

        :param train_data: Time series data used for training.
        """

        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        self.model = CATCHModel(self.config)
        self.model.to(self.device)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        train_steps = len(self.train_data_loader)
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]

        self.optimizer = torch.optim.Adam(main_params,
                                          lr=self.config.lr)
        self.optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=self.config.Mlr)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )

        schedulerM = lr_scheduler.OneCycleLR(
            optimizer=self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.Mlr,
        )

        time_now = time.time()

        for epoch in range(self.config.num_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            self.model.train()

            step = min(int(len(self.train_data_loader) / 10), 100)
            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                input = input.float().to(self.device)

                output, output_complex, dcloss = self.model(input)

                output = output[:, :, :]

                rec_loss = self.criterion(output, input)

                norm_input = self.model.revin_layer(input, 'transform')
                auxi_loss = self.auxi_loss(output_complex, norm_input)

                loss = rec_loss + config.dc_lambda * dcloss + config.auxi_lambda * auxi_loss

                train_loss.append(loss.item())

                if (i + 1) % step == 0:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | training time loss: {2:.7f} | training fre loss: {3:.7f} | training dc loss: {4:.7f}".format(
                            i + 1, epoch + 1, rec_loss.item(), auxi_loss.item(), dcloss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss
                )
            )

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(
                    "\t testing time loss: {0} | \n testing fre loss: {1}".format(
                        temp_score.detach().cpu().numpy()[0,:5], freq_score.detach().cpu().numpy()[0,:5]
                    )
                )

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)

                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(
                    "\t testing time loss: {0} | \n\t testing fre loss: {1}".format(
                        temp_score.detach().cpu().numpy()[0,:5], freq_score.detach().cpu().numpy()[0,:5]
                    )
                )

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy