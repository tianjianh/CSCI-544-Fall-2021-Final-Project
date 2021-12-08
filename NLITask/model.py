import torch
import torch.nn as nn

from module import *


def rep_matrix(length, device):
	batch_size = len(length)
	seq_len = torch.max(length)

	rep_matrix = torch.FloatTensor(batch_size, seq_len).to(torch.device(device))
	rep_matrix.data.fill_(1)

	for i in range(batch_size):
		rep_matrix[i, length[i]:] = 0

	return rep_matrix.unsqueeze_(-1)


class NLItask(nn.Module):

	def __init__(self, args, data):
		super(NLItask, self).__init__()

		self.class_size = args.class_size
		self.dropout = args.dropout
		self.d_e = args.d_e
		self.d_ff = args.d_ff
		self.device = args.device

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
		# initialize word embedding with GloVe
		self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)

		# fine-tune the word embedding
		self.word_emb.weight.requires_grad = False

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)



		self.sentence_encoder = Hyp_pre_Encoder(args)

		self.fc = nn.Linear(args.d_e * 4 * 4, args.d_e)
		self.fc_out = nn.Linear(args.d_e, args.class_size)

		self.layer_norm = nn.LayerNorm(args.d_e)
		self.dropout = nn.Dropout(args.dropout)
		self.relu = nn.ReLU()

	def forward(self, batch):
		premise, pre_len = batch.premise
		hypothesis, hypo_len = batch.hypothesis

		#Size
		# (batch_size, seq_len, word_dimsion)
		pre_x = self.word_emb(premise)
		hypo_x = self.word_emb(hypothesis)

		# (batch_size, seq_len, 1)
		pre_matrix = rep_matrix(pre_len, self.device)
		hypo_matrix = rep_matrix(hypo_len, self.device)

		# (batch, seq_len, 4 * d_e)
		pre_sen = self.sentence_encoder(pre_x, pre_matrix)
		hypo_sen = self.sentence_encoder(hypo_x, hypo_matrix)

		# (batch, seq_len, 4 * 4 * d_e)
		s = torch.cat([pre_sen, hypo_sen, (pre_sen - hypo_sen).abs(), pre_sen * hypo_sen], dim=-1)

		s = self.dropout(s)
		outs = self.relu(self.layer_norm(self.fc(s)))
		outs = self.dropout(outs)
		outs = self.fc_out(outs)

		return outs

