import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn
#正则化，归一化用了一下
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight=self.affine_weight.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.affine_bias=self.affine_bias.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
from tkinter import _flatten
import torch.nn.functional as F
# try:
#     from tkinter import _flatten
# except ImportError:
#     _flatten = lambda l: [item for sublist in l for item in sublist]
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size) # 第二个全连接层


    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        #x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        return x




class LFTSAD_(nn.Module):
    def __init__(self, win_size, enc_in, patch_seq,seq_size, c_out, d_model=256,patch_size=[3,5,7], channel=55, d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(LFTSAD_, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size#list
        self.channel = channel
        self.win_size = win_size
        self.patch_seq=patch_seq###list
        self.seq_size=seq_size## int

        mlp_num_input_size = [self.win_size // patchsize - 1 for patchsize in self.patch_size]
        mlp_num_seq_input_size = [self.win_size // patch_seq - self.seq_size for patch_seq in self.patch_seq]

        # Initialize MLP layers
        self.mlp_size = nn.ModuleList(
            MLP(patchsize - 1, d_model, 1) for patchsize in self.patch_size)

        self.mlp_num = nn.ModuleList(
            MLP(input_size, d_model, 1) for input_size in mlp_num_input_size)

        self.mlp_size_seq = nn.ModuleList(
            MLP((patch_seq - 1) * self.seq_size, d_model, self.seq_size) for patch_seq in self.patch_seq)

        self.mlp_num_seq = nn.ModuleList(
            MLP(input_size, d_model, self.seq_size) for input_size in mlp_num_seq_input_size)



    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel   128 100 51
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')

        
        ###########点级别
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x #128,100,51

            #预处理size
            result=[]
            x_patch_size= rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_size = rearrange(x_patch_size, 'b m (p n) -> (b m) p n', p=patchsize)  # 6258 5 20
            all_indices = list(range(patchsize))
            for i in range(patchsize):  ###排除重构点
                indices = [idx for idx in all_indices if idx != i]
                temp1=x_patch_size[:,indices,:].permute(0,2,1)
                result.append(temp1)


            x_patch_size = torch.cat(result, axis=1).permute(1,0,2)  # 8*105 1 35
            x_patch_size = self.mlp_size[patch_index](x_patch_size).squeeze(-1).permute(1,0).reshape(-1,self.channel,self.win_size).permute(0,2,1)

            num = self.win_size // patchsize
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize)  # 6258 5 20
            result = []
            for i in range (L):
                part = torch.cat((x_patch_num[:,i%patchsize,0:i//patchsize],x_patch_num[:,i%patchsize,i//patchsize+1:num]),dim=1)
                result.append(part)
            x_patch_num = torch.cat(result, axis=0)
            x_patch_num = self.mlp_num[patch_index](x_patch_num)
            x_patch_num = x_patch_num.reshape(B,M,L).permute(0,2,1)#B L M

            series_patch_mean.append(x_patch_size), prior_patch_mean.append(x_patch_num)

        series_patch_mean = list(_flatten(series_patch_mean)) #3
        prior_patch_mean = list(_flatten(prior_patch_mean)) #3

        series_patch_seq = []
        prior_patch_seq = []
        ###########子序列
        for patch_index, patchsize in enumerate(self.patch_seq):
            x_patch_size, x_patch_num = x, x  # 128,100,51

            # 预处理size
            result = []
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_size = rearrange(x_patch_size, 'b m (p n s) -> (b m) p n s', p=patchsize,s=self.seq_size)  # 6258 5 20
            all_indices = list(range(patchsize))
            for i in range(patchsize):
                indices = [idx for idx in all_indices if idx != i]
                temp1 =rearrange( x_patch_size[:, indices, :,:].permute(0,2,1,3),'a b c d -> a b (c d) ')
                result.append(temp1)

            x_patch_size = torch.cat(result, axis=1)  # 8*105 1 35
            x_patch_size = rearrange( self.mlp_size_seq[patch_index](x_patch_size),'(a b) c d  -> a b (c d)',
                                      b=self.channel).permute(0,2,1)

            num = self.win_size // patchsize
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_num = rearrange(x_patch_num, 'b m (p n s) -> (b m) p n  s', p=patchsize,
                                    s=self.seq_size)  # 6258 5 20
            result = []
            all_indices = list(range(x_patch_num.shape[2]))
            for i in range(x_patch_num.shape[2]):
                indices = [idx for idx in all_indices if idx != i]
                temp1 = rearrange(x_patch_num[:, :, indices, :], 'a b c d ->a b (c d) ')
                result.append(temp1)
            x_patch_num = torch.cat(result, axis=1)
            x_patch_num = self.mlp_num_seq[patch_index](x_patch_num)
            x_patch_num = rearrange(rearrange(x_patch_num,'(a b)  (c  d) e  -> a  b  c  d  e', b=self.channel, d=self.patch_seq[0]).permute(0,1,3,2,4),
                                    ' a b c d e -> a b (c d e)').permute(0,2,1)

            series_patch_seq.append(x_patch_size), prior_patch_seq.append(x_patch_num)

        series_patch_seq = list(_flatten(series_patch_seq))  # 3
        prior_patch_seq = list(_flatten(prior_patch_seq))  # 3

            
        if self.output_attention:
            return series_patch_mean, prior_patch_mean,series_patch_seq, prior_patch_seq
        else:
            return None
        

from .base import BaseDetector

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
from einops import rearrange
import warnings
import pandas as pd
warnings.filterwarnings('ignore')





def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = None
        self.vali_loader = None
        self.test_loader = None
        self.thre_loader = None
        self.sw_max_mean = self.sw_max_mean
        print('sw_max_mean: ', self.sw_max_mean)
        self.sw_loss = self.sw_loss
        self.p_seq = self.p_seq
        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep= nn.MSELoss(reduction='none')


    def build_model(self):
        self.model = LFTSAD_(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c,
                                d_model=self.d_model, patch_size=self.patch_size, channel=self.input_c,
                                patch_seq=self.patch_seq,seq_size=self.seq_size)

        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self):

        time_now = time.time()

        train_steps = len(self.train_loader) #3866

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device) #(128,100,51)
                series, prior, series_seq, prior_seq = self.model(input)

                loss = 0.0
                for u in range(len(prior)):
                    if (self.sw_loss == 0):
                        loss += (self.p_seq * self.criterion(series_seq[u], prior_seq[u]) + (1 - self.p_seq) * self.criterion(
                            series[u], prior[u]))
                    else:
                        loss += (self.p_seq * self.criterion(series_seq[u], prior_seq[u]) + (1 - self.p_seq) * self.criterion(
                            series[u], prior[u]))


                loss = loss / len(prior)
                # loss = revin_layer(loss, 'norm')
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    @torch.no_grad()
    def test(self):
        self.model.eval()

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior, series_seq, prior_seq = self.model(input)

            loss = 0
            for u in range(len(prior)):

                if (self.sw_loss == 0):
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])
                else:
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])

            if (self.sw_max_mean == 0):
                loss = torch.mean(loss, dim=-1)
            else:
                loss, _ = torch.max(loss, dim=-1)

            metric = torch.softmax(loss, dim=-1)
            # metric = loss
            cri = metric.detach().cpu().numpy()[:,-1]
            attens_energy.append(cri)

        if (len(attens_energy) == 0):
            print("win_size * batchsize的乘积过大，请适当调小")
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        test_labels = []
        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior, series_seq, prior_seq = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            test_labels.append(labels)
            loss = 0
            for u in range(len(prior)):

                if (self.sw_loss == 0):
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])
                else:
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])

            if (self.sw_max_mean == 0):
                loss = torch.mean(loss, dim=-1)
            else:
                loss, _ = torch.max(loss, dim=-1)
            metric = torch.softmax(loss, dim=-1)
            cri = metric.detach().cpu().numpy()[:,-1]
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        test_energy = np.array(attens_energy)

        


        return test_energy
        
    

from torch.utils.data import DataLoader, TensorDataset
from .feature import Window

class MyArgumentParser():
    
    def add_argument(self, name, type=None, default=None, help=None, action=None):
        if action == 'store_true':
            setattr(self, name.lstrip('--'), False)
        else:
            setattr(self, name.lstrip('--'), type(default))

    def parse_args(self):
        return self
    
import math

class LFTSAD_Detector(BaseDetector):

    def config_init(self):
        import argparse
        import json

        parser = MyArgumentParser()

        parser.add_argument('--win_size', type=int, default=100)#窗口大小，减小窗口大小可能有助于捕捉更精细的特征，而增大窗口大小则可能有助于捕捉更全局的特征。
        parser.add_argument('--patch_size', type=str, default='[10]')#
        parser.add_argument('--patch_seq', type=str, default='[5]')#
        parser.add_argument('--seq_size', type=int, default=5)#
        parser.add_argument('--num_epochs', type=int, default=1)#训练轮次，防止过拟合
        parser.add_argument('--batch_size', type=int, default=128)#批量大小，较小使模型更快但不稳定。较大更加稳定但较慢
        parser.add_argument('--input_c', type=int, default=55)#输入通道数
        parser.add_argument('--d_model', type=int, default=128)#模型维度，增加或减少这个值可能会改变模型的容量和复杂度
        parser.add_argument('--anormly_ratio', type=float, default=0.82)#异常检测的比例
        parser.add_argument('-sw_max_mean', type=int, default=0, help='0:mean , 1:max')
        parser.add_argument('-sw_loss', type=int, default=0, help='0:mse  , 1:mae')
        parser.add_argument('-p_seq', type=float, default=0, help='  点级别和子序列比例  ')

        parser.add_argument('-min_size', type=int, default=256)
        parser.add_argument('--output_c', type=int, default=1)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--loss_fuc', type=str, default='MSE')
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # Default
        parser.add_argument('--index', type=int, default=137)

        config = parser.parse_args()

        # Convert JSON string to list of ints
        config.patch_size = [int(x) for x in json.loads(config.patch_size)]
        config.patch_seq = [int(x) for x in json.loads(config.patch_seq)]

        config.output_c = config.input_c
        config.min_size = config.batch_size

        config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False

        self.config = config


    def __init__(self, win_size=100, 
                   stride=1,
                 batch_size=128,
                 epochs=1,
                 lr=1e-4
                 ):
        self.win_size = win_size
        self.stride = stride
        self.batch_size = batch_size
        self.epochs = epochs
        self.config_init()
        self.config.win_size = win_size
        self.config.batch_size = batch_size
        self.config.num_epochs = epochs
        self.config.lr = lr


    def fit(self, X, y=None):
        n_samples, feature_dim = X.shape
        self.config.input_c = feature_dim
        self.config.output_c = feature_dim
        self.solver = Solver(vars(self.config))

        Xw = Window(window=self.win_size, stride=self.stride).convert(X)
        win = self.win_size
        X_seq = Xw.reshape(-1, win, feature_dim)
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(torch.zeros((X_seq.shape[0],X_seq.shape[1])), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.solver.train_loader = loader
        self.solver.train()

    def decision_function(self, X):
        n_samples, feature_dim = X.shape
        print('Feature dim: ', feature_dim, 'Win size: ', self.win_size, 'N samples: ', n_samples)
        Xw = Window(window=self.win_size, stride=1).convert(X)
        win = self.win_size
        X_seq = Xw.reshape(-1, win, feature_dim)
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(torch.zeros(X_seq.shape), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.solver.thre_loader = loader
        scores = self.solver.test()

        if scores.shape[0] < len(X):
            # self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
            #         list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
            scores = np.array([scores[0]]*math.ceil((self.win_size-1)/2) + 
                    list(scores) + [scores[-1]]*((self.win_size-1)//2))

        return scores

        
