import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

torch.manual_seed(1)

import numpy as np

class SCTM(nn.Module):
	def __init__(self, seq_dim, hidden_dim):
		super(SCTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.seq_dim = seq_dim

		self.dft_real_weight = nn.Parameter(torch.Tensor(seq_dim, hidden_dim), requires_grad=True)
		self.dft_imag_weight = nn.Parameter(torch.Tensor(seq_dim, hidden_dim), requires_grad=True)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.dft_real_weight)
		nn.init.xavier_uniform_(self.dft_imag_weight)

		# fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dft_real_weight)
		# bound = 1 / math.sqrt(fan_in)
		# nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		real_part = torch.cos(torch.matmul(x, self.dft_real_weight))
		imag_part = torch.sin(torch.matmul(x, self.dft_imag_weight))
		# output = torch.cat((real_part, imag_part), dim=-1)
		# output = output[:, :, ::2]
		output = 0.3 * real_part + 0.7 * imag_part
		return output

class JCLinear(nn.Module):
	def __init__(self, seq_dim, hidden_dim):
		super(JCLinear, self).__init__()
		self.hidden_dim = hidden_dim
		self.seq_dim = seq_dim

		self.dft_real_weight = nn.Parameter(torch.Tensor(seq_dim, hidden_dim))
		self.dft_imag_weight = nn.Parameter(torch.Tensor(seq_dim, hidden_dim))

		# self.bias = nn.Parameter(torch.Tensor(seq_dim))

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.dft_real_weight)
		nn.init.xavier_uniform_(self.dft_imag_weight)

		# fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dft_real_weight)
		# bound = 1 / math.sqrt(fan_in)
		# nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x, y):
		real_part = torch.cos(torch.matmul(x, self.dft_real_weight))
		imag_part = torch.sin(torch.matmul(y, self.dft_imag_weight))
		for i in range(10):
			output = torch.cat((real_part, imag_part[i:i+1, :, :]), dim=-1)
			output = output[:, :, ::2]
			return output
		# output = torch.cat((real_part, imag_part), dim=-1)
		# output = output[:, :, ::2]
		# return output