#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-21 下午7:29
"""一、CNN-RNN 网络架构  三层CNN
1.第一层网络：16个卷积核，尺寸为5*5，步长为2；2*2最大池化；tanh激活函数
2.第二层网络：64个卷积核，尺寸为5*5，步长为2；2*2最大池化；tanh激活函数
3.第三层网络：64个卷积核，尺寸为5*5，步长为2；tanh激活函数
4.0.5的dropout
5.128个元素的FC全连接层

二、空间金字塔池化 8*8，4*4,2*2,1*1

"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from models.attention import *


class Net(nn.Module):
	def __init__(self, nFilter1, nFilter2, nFilter3, num_person_train, dropout=0.0, num_features=0, seq_len=0, batch=0):
		super(Net, self).__init__()
		self.batch = batch
		self.seq_len = seq_len
		self.num_person_train = num_person_train
		self.dropout = dropout  # 随机失活的概率，0-1
		self.num_features = num_features  # 输出的特征维度 128
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.nFilters = [nFilter1, nFilter2, nFilter3]  # 初始化每一层的卷积核个数
		self.filter_size = [5, 5, 5]  # 卷积核尺寸

		self.poolsize = [2, 2, 2]  # 最大池化的尺寸
		self.stepsize = [2, 2, 2]  # 池化步长
		self.padDim = 4  # 零填充
		self.input_channel = 5  # 3img + 2optical flow

		# 构建卷积层，nn。Conv2d(输入通道，卷积核的个数，卷积核尺寸，步长，零填充)
		self.conv1 = nn.Conv2d(self.input_channel, self.nFilters[0], self.filter_size[0], stride=1, padding=self.padDim)
		self.conv2 = nn.Conv2d(self.nFilters[0], self.nFilters[1], self.filter_size[1], stride=1, padding=self.padDim)
		self.conv3 = nn.Conv2d(self.nFilters[1], self.nFilters[2], self.filter_size[2], stride=1, padding=self.padDim)

		# 构建最大池化层
		self.pooling1 = nn.MaxPool2d(self.poolsize[0], self.stepsize[0])
		self.pooling2 = nn.MaxPool2d(self.poolsize[1], self.stepsize[1])

		# tanh激活函数
		self.tanh = nn.Tanh()

		# 4个空间金字塔池化层
		self.pool1 = nn.Sequential(nn.AdaptiveMaxPool2d((8, 8)))
		self.pool2 = nn.Sequential(nn.AdaptiveMaxPool2d((4, 4)))
		self.pool3 = nn.Sequential(nn.AdaptiveMaxPool2d((2, 2)))
		self.pool4 = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)))

		# FC层
		n_fully_connected = 32 * (8*8 + 4*4 + 2*2 + 1*1)  # 根据图片尺寸修改

		self.seq2 = nn.Sequential(
			nn.Dropout(self.dropout),
			nn.Linear(n_fully_connected, self.num_features)
		)

		# rnn层
		# self.rnn = nn.LSTM(input_size=self.num_features, hidden_size=self.num_features, num_layers=1, batch_first=True, dropout=self.dropout)

		self.rnn = nn.RNN(input_size=self.num_features, hidden_size=self.num_features)
		self.hid_weight = nn.Parameter(
			nn.init.xavier_uniform_(torch.Tensor(1, self.seq_len, self.num_features).type(
				torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
			), gain=np.sqrt(2.0)), requires_grad=True)

		# 注意力层
		self.attention = Attention(self.num_features)
		self.add_module('attention', self.attention)

		# final full connectlayer
		self.final_FC = nn.Linear(self.num_features, self.num_person_train)

	def build_net(self, input1, input2):
		seq1 = nn.Sequential(
			self.conv1, self.tanh, self.pooling1,
			self.conv2, self.tanh, self.pooling2,
			self.conv3, self.tanh,
		)
		# b = input1.size(0)  # batch的大小    # 测试时为片段的长度
		# n = input1.size(1)  # 1个batch中图片的数目
		# input1 = input1.view(b*n, input1.size(2), input1.size(3), input1.size(4))  # 测试时torch.Size([144, 5, 256, 128])
		# input2 = input2.view(b*n, input2.size(2), input2.size(3), input2.size(4))
		inp1_seq1_out = seq1(input1)  # torch.Size([16, 32, 35, 19])
		inp2_seq1_out = seq1(input2)  # 经过卷积层后的输出

		inp1_spp_out = self.spatial_pooling(inp1_seq1_out)  # torch.Size([16, 2720])
		inp2_spp_out = self.spatial_pooling(inp2_seq1_out)

		inp1_seq2_out = self.seq2(inp1_spp_out).unsqueeze(0)  # torch.Size([1, 16, 128])
		inp2_seq2_out = self.seq2(inp2_spp_out).unsqueeze(0)  # 经过fc层的输出  torch.Size([1, 16, 128])

		inp1_rnn_out, hn1 = self.rnn(inp1_seq2_out)
		inp2_rnn_out, hn2 = self.rnn(inp2_seq2_out)   # torch.Size([1, 16, 128])
		feature_p = inp1_rnn_out.squeeze()  # torch.Size([16, 128])
		feature_g = inp2_rnn_out.squeeze()

		feature_p, feature_g = self.attention(feature_p, feature_g)  # torch.Size([1, 128])

		# 分类
		identity_p = self.final_FC(feature_p)  # 身份特征 torch.Size([1, 89])
		identity_g = self.final_FC(feature_g)
		return feature_p, feature_g, identity_p, identity_g

	def spatial_pooling(self, inputs):
		out1 = self.pool1(inputs)
		out1_shape = out1.shape  # torch.Size([128, 32, 8, 8])
		sec1_dim = out1_shape[-1] * out1_shape[-2] * out1_shape[-3]  # 2048
		out1 = out1.contiguous().view(-1, sec1_dim)  # torch.Size([128, 2048])

		out2 = self.pool2(inputs)
		out2_shape = out2.shape  # torch.Size([128, 32, 4, 4])
		sec2_dim = out2_shape[-1] * out2_shape[-2] * out2_shape[-3]  # 512
		out2 = out2.contiguous().view(-1, sec2_dim)  # torch.Size([128, 512])

		out3 = self.pool3(inputs)
		out3_shape = out3.shape  # torch.Size([128, 32, 2, 2])
		sec3_dim = out3_shape[-1] * out3_shape[-2] * out3_shape[-3]  # 128
		out3 = out3.contiguous().view(-1, sec3_dim)  # torch.Size([128, 128])

		out4 = self.pool4(inputs)
		out4_shape = out4.shape  # torch.Size([128, 32, 1, 1])
		sec4_dim = out4_shape[-1] * out4_shape[-2] * out4_shape[-3]  # 32
		out4 = out4.contiguous().view(-1, sec4_dim)  # torch.Size([128, 32])

		outputs = torch.cat((out1, out2, out3, out4), 1)  # torch.Size([128, 2720])
		return outputs

	def forward(self, input1, input2):
		feature_p, feature_g, identity_p, identity_g = self.build_net(input1, input2)
		return feature_p, feature_g, identity_p, identity_g

	def initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.xavier_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				torch.nn.init.normal_(m.weight.data, 0, 0.01)
				m.bias.data.zero_()


class Criterion(nn.Module):
	def __init__(self, hinge_margin=2):
		super(Criterion, self).__init__()
		self.hinge_margin = hinge_margin
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def forward(self, feature_p, feature_g, identity_p, identity_g, target):
		log_soft = nn.LogSoftmax(1)
		lsoft_p = log_soft(identity_p)
		lsoft_g = log_soft(identity_g)

		dist = nn.PairwiseDistance(p=2)
		pair_dist = dist(feature_p, feature_g)  # 欧几里得距离

		# 1.折页损失
		hing = nn.HingeEmbeddingLoss(margin=self.hinge_margin, reduce=False)
		label0 = torch.tensor(target[0]).type(
			torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
		hing_loss = hing(pair_dist, label0)

		# 2.交叉熵损失
		nll = nn.NLLLoss()
		label1 = torch.tensor([target[1]]).type(
			torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
		label2 = torch.tensor([target[2]]).type(
			torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
		loss_p = nll(lsoft_p, label1)
		loss_g = nll(lsoft_g, label2)

		# 3.损失求和
		total_loss = hing_loss + loss_p + loss_g
		# mean_loss = torch.mean(total_loss)
		# loss = torch.sum(total_loss)

		return total_loss



