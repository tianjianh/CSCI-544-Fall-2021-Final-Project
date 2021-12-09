import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import math


# Masked softmax
def m_softmax(vec, mask_matrix, dim=1):
	mask_vector = vec * mask_matrix.float()

	max_vector = torch.max(mask_vector, dim=dim, keepdim=True)[0]

	expression = torch.exp(mask_vector - max_vector)

	masked_exp = expression * mask_matrix.float()

	sums = masked_exp.sum(dim, keepdim=True)

	zeros = (sums == 0)

	sums += zeros.float()

	return masked_exp / (sums + 1e-20)

# Directional mask
def directional_matrix(direction, sentence_len, device):
	matrix = torch.FloatTensor(sentence_len, sentence_len).to(torch.device(device))

	matrix.data.fill_(1)

	if direction == 'forward':
		matrix = torch.tril(matrix, diagonal=-1)

	else:
		matrix = torch.triu(matrix, diagonal=1)

	matrix.unsqueeze_(0)

	return matrix

# Representation mask for sentences of variable lengths
def sentence_fixed_matrix(mask_matrix):
	batch_size, sen_len, _ = mask_matrix.size()

	m1 = mask_matrix.view(batch_size, sen_len, 1)

	m2 = mask_matrix.view(batch_size, 1, sen_len)

	mask_representation = torch.mul(m1, m2)

	return mask_representation

# Distance mask
def distance_matrix(sen_len, device):
	dis_matrix = torch.FloatTensor(sen_len, sen_len).to(torch.device(device))

	for i in range(sen_len):
		for j in range(sen_len):
			dis_matrix[i, j] = -abs(i-j)

	dis_matrix.unsqueeze_(0)

	return dis_matrix


class single_Attention(nn.Module):

	def __init__(self, d_model, direction, alpha, device='cuda:0', args):
		super(single_Attention, self).__init__()

		self.direction = direction

		self.device = device

		self.alpha = alpha

		self.scaling_factor = Variable(torch.Tensor([math.pow(d_model, 0.5)]), requires_grad=False).cuda()

		self.softmax = nn.Softmax(dim=2)


	def forward(self, q, k, v, repsentation_mask):
		batch_size, seq_len, d_model = q.size()
		
		multi_att = torch.bmm(q, k.transpose(1, 2)) / self.scaling_factor

		direct_matrix = directional_matrix(self.direction, seq_len, self.device)

		rep_matrix = sentence_fixed_matrix(repsentation_mask)



		#There is where we added distance matrix in it.
		if args.poe = True:

			dist_matrix = distance_matrix(seq_len, self.device)
			mask = rep_matrix * direct_matrix
			multi_att += self.alpha * dist_matrix

		multi_att = m_softmax(multi_att, mask, dim=2)

		out = torch.bmm(multi_att, v)

		return out, multi_att


class MultiHeadAttention(nn.Module):

	def __init__(self, args, direction):
		super(MultiHeadAttention, self).__init__()

		self.n_head = args.num_heads
		self.d_k = args.d_e // args.num_heads
		self.d_v = args.d_e // args.num_heads
		self.d_model = args.d_e

		self.weight_qs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
		self.weight_ks = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
		self.weight_vs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_v))

		self.attention = single_Attention(self.d_model, direction, args.alpha, device=args.device, args)
		self.layer_norm = nn.LayerNorm(int(self.d_k))
		self.layer_norm2 = nn.LayerNorm(self.d_model)

		self.l1 = nn.Linear(self.n_head * self.d_v, self.d_model)

		self.dropout = nn.Dropout(args.dropout)

		#Initialize, can use some other distribution
		init.xavier_normal_(self.weight_qs)
		init.xavier_normal_(self.weight_ks)
		init.xavier_normal_(self.weight_vs)


	def forward(self, q, k, v, repsentation_matrix):
		n_head = self.n_head

		mb_size, len_q, d_model = q.size()
		mb_size, len_k, d_model = k.size()
		mb_size, len_v, d_model = v.size()

		q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
		k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
		v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)

		q_s = self.layer_norm(torch.bmm(q_s, self.weight_qs).view(-1, len_q, self.d_k))
		k_s = self.layer_norm(torch.bmm(k_s, self.weight_ks).view(-1, len_k, self.d_k))
		v_s = self.layer_norm(torch.bmm(v_s, self.weight_vs).view(-1, len_v, self.d_v))

		rep_matrix = repsentation_matrix.repeat(n_head, 1, 1).view(-1, len_q, 1)

		outs, attns = self.attention(q_s, k_s, v_s, rep_matrix)

		outs = torch.cat(torch.split(outs, mb_size, dim=0), dim=-1)

		outs = self.layer_norm2(self.l1(outs))

		outs = self.dropout(outs)

		return outs


class FusionGate(nn.Module):

	def __init__(self, d_e, dropout=0.1):

		super(FusionGate, self).__init__()

		self.weight_s = nn.Parameter(torch.FloatTensor(d_e, d_e))
		self.weight_h = nn.Parameter(torch.FloatTensor(d_e, d_e))
		self.b = nn.Parameter(torch.FloatTensor(d_e))

		#Initialize
		init.xavier_normal_(self.weight_s)
		init.xavier_normal_(self.weight_h)
		init.constant_(self.b, 0)

		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_e)


	def forward(self, s, h):

		s_f = self.layer_norm(torch.matmul(s, self.weight_s))
		h_f = self.layer_norm(torch.matmul(h, self.weight_h))

		f = self.sigmoid(self.dropout(s_f + h_f + self.b))

		outs = f * s_f + (1 - f) * h_f

		return self.layer_norm(outs)


class PositionwiseFeedForward(nn.Module):

	def __init__(self, d_h, d_in_h, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()

		self.ffd = args.ffd
		if args.ffd == "conv1d":

			self.cov_1 = nn.Conv1d(d_h, d_in_h, 1)  # position-wise
			self.cov_2 = nn.Conv1d(d_in_h, d_h, 1)  # position-wise

		else:

			self.l1 = nn.Linear(d_h, d_in_h)
			self.l2 = nn.Linear(d_in_h, d_h)

		self.layer_norm = nn.LayerNorm(d_h)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()


	def forward(self, x):

		if self.ffd == "conv1d":
			out = self.relu(self.cov_1(x.transpose(1, 2)))

			out = self.cov_2(out).transpose(2, 1)

			out = self.dropout(out)

		else:

			out = self.relu(self.l1(x))

			out = self.l2(out)

			out = self.dropout(out)


		return self.layer_norm(out + x)


class Half_Encoder(nn.Module):

	def __init__(self, args, direction):
		super(Half_Encoder, self).__init__()

		self.attn_layer = MultiHeadAttention(args, direction)

		self.fusion_gate = FusionGate(args.d_e, args.dropout)

		self.feed_forward = PositionwiseFeedForward(args.d_e, args.d_ff, args.dropout, args)


	def forward(self, x, rep_mask):
		outs = self.attn_layer(x, x, x, rep_mask)

		outs = self.fusion_gate(x, outs)

		outs = self.feed_forward(outs)

		return outs


class Source2Token(nn.Module):

	def __init__(self, d_h, dropout=0.1):
		super(Source2Token, self).__init__()

		self.d_h = d_h
		self.dropout_rate = dropout

		self.fc1 = nn.Linear(d_h, d_h)
		self.fc2 = nn.Linear(d_h, d_h)

		self.elu = nn.ELU()
		self.softmax = nn.Softmax(dim=1)
		self.layer_norm = nn.LayerNorm(d_h)


	def forward(self, x, rep_mask):
		out = self.elu(self.layer_norm(self.fc1(x)))
		out = self.layer_norm(self.fc2(out))

		out = m_softmax(out, rep_mask, dim=1)
		out = torch.sum(torch.mul(x, out), dim=1)

		return out


class Hyp_pre_Encoder(nn.Module):

	def __init__(self, args):
		super(Hyp_pre_Encoder, self).__init__()

		# forward and backward transformer block
		self.forward_left = Half_Encoder(args, direction='forward')
		self.backward_right = Half_Encoder(args, direction='backword')

		# Multi-dimensional source2token self-attention
		self.source2token = Source2Token(d_h=2 * args.d_e, dropout=args.dropout)


	def forward(self, inputs, rep_matrix):
		batch, seq_len, d_e = inputs.size()

		left_encod = self.forward_left(inputs, rep_matrix)
		right_encod = self.backward_right(inputs, rep_matrix)

		u = torch.cat([left_encod, right_encod], dim=-1)

		pooling = nn.MaxPool2d((seq_len, 1), stride=1)

		pool_v2 = pooling(u * rep_matrix).view(batch, -1)

		source2token_result = self.source2token(u, rep_matrix)

		outs = torch.cat([source2token_result, pool_v2], dim=-1)

		return outs







