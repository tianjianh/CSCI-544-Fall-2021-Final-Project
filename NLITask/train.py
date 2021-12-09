import argparse
import copy
import os
import torch

from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model import NLItask
from data import dataset_import
from test import test


def train(args, data):
	model = NLItask(args, data)
	model.to(torch.device(args.device))

	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = optim.Adam(parameters, lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss()

	writer = SummaryWriter(log_dir='runs/' + args.model_time)

	model.train()
	acc, loss, size, last_epoch = 0, 0, 0, -1
	max_dev_acc, max_test_acc = 0, 0

	iterator = data.train_iter

	for epoch in range(args.epoch):

		print('epoch:', epoch + 1)

		for i, batch in enumerate(iterator):
			pred = model(batch)
			optimizer.zero_grad()
			batch_loss = criterion(pred, batch.label)
			loss += batch_loss.item()
			batch_loss.backward()
			optimizer.step()

			_, pred = pred.max(dim=1)
			acc += (pred == batch.label).sum().float()
			size += len(pred)

			if (i + 1) % args.print_freq == 0:
				# print(iterator.epoch)
				acc /= size

				#Here We use cpu, if you have resource, you can use cuda
				acc = acc.cpu().item()

				dev_loss, dev_acc = test(model, data, mode='dev')
				test_loss, test_acc = test(model, data)
				c = (i + 1) // args.print_freq

				writer.add_scalar('loss/train', loss, c)
				writer.add_scalar('acc/train', acc, c)
				writer.add_scalar('loss/dev', dev_loss, c)
				writer.add_scalar('acc/dev', dev_acc, c)
				writer.add_scalar('loss/test', test_loss, c)
				writer.add_scalar('acc/test', test_acc, c)

				print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
					  f' / train acc: {acc:.3f} / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f}')

				#Use the dev to save the best model
				if dev_acc > max_dev_acc:
					max_dev_acc = dev_acc
					max_test_acc = test_acc
					best_model = copy.deepcopy(model)

				acc, loss, size = 0, 0, 0
				model.train()

	writer.close()
	print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

	return best_model


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', default=64, type=int)
	parser.add_argument('--data-type', default='SNLI')#####Here, can be replaced by other dataset
	parser.add_argument('--dropout', default=0.1, type=float)
	parser.add_argument('--epoch', default=20, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--learning-rate', default=0.001, type=float)
	parser.add_argument('--print-freq', default=1000, type=int)  ####Guarantee that we have output
	parser.add_argument('--word-dim', default=300, type=int)
	parser.add_argument('--num-heads', default=5, type=int)
	parser.add_argument('--d-ff', default=300 * 4, type=int)
	parser.add_argument('--alpha', default=1.5, type=float)
	parser.add_argument('--ffd', default="conv1d", type=str)
	parser.add_argument('--poe', default=True, type=bool)
	parser.add_argument('--alpha', default=1.5, type=float)



	args = parser.parse_args()

	if args.gpu > -1:
		setattr(args, 'device', "cuda:0")
	else:
		setattr(args, 'device', "cpu")

	print('loading dataset for train...')
	data = dataset_import(args)


	setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
	setattr(args, 'class_size', len(data.LABEL.vocab))
	setattr(args, 'model_running_time', strftime('%H-%M-%S', gmtime()))
	setattr(args, 'word_dimension', args.word_dim)

	print('training start')
	best_model = train(args, data)

	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')
	torch.save(best_model.state_dict(), f'saved_models/NLITASK_{args.data_type}_{args.model_time}.pt')

	print('Finished!')


if __name__ == '__main__':
	main()
