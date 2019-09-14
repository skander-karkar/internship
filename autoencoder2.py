import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders import dataloaders
from utils2 import *
from torchsummary import summary
import time, numpy as np, argparse, os
from functools import partial

class Autoencoder(nn.Module):
    def __init__(self, in_channels, filters, ds, bn):
        super(Autoencoder, self).__init__()
        self.encoder, self.decoder = create_autoencoder(in_channels, filters, ds, bn)
        initialization = partial(initialize, "orthogonal", 1)
        self.apply(initialization)
    def forward(self,x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z

def train_and_save(filters, ds = True, bn = False, pen = 0, lmbda = 0, nepochs = 20):
	name = 'AE2-' + ('ds' if ds else '') + ('bn' if bn else '') + str(filters) + ('pen' + str(pen) if pen else '') + ('lmbda' + str(int(1 / lmbda)) if pen else '')
	print('-' * 60, 'Running:', name)
	autoencoder = Autoencoder(data_shape[1], filters, ds, bn)
	autoencoder.to(device)
	print(autoencoder)
	summary(autoencoder, data_shape[1:])
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(autoencoder.parameters())
	t = time.time() 
	for epoch in range(1, nepochs + 1):
		autoencoder.train()
		train_loss = 0
		for x, _ in trainloader:
			optimizer.zero_grad()
			x = x.to(device)
			y, z = autoencoder(x)
			loss = criterion(y, x) 
			if pen == 1:
				loss += lmbda * torch.norm(z, 1)
			elif pen == 2:
				loss += lmbda * torch.norm(autoencoder.encoder[2].weight, 1) 
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		autoencoder.eval()
		test_loss = 0
		idx_batch = np.random.choice(ntest)
		for i, (x, _) in enumerate(testloader):
			x = x.to(device)
			y, z = autoencoder(x)
			loss = criterion(y, x)
			test_loss += loss.item()
			if i == idx_batch:
				idx_images = np.random.choice(x.size()[0], 5, replace = False)
				x_ = x.cpu().detach().numpy().copy()[idx_images, :, :, :]
				y_ = y.cpu().detach().numpy().copy()[idx_images, :, :, :]
				show_autoencoder_images(x_, y_, mean, std, os.path.join(examples_folder, name + 'epoch' + str(epoch) + '.png'))
		print('[Epoch {}/{}] Train loss : {:.4f} | Test loss : {:.4f} | Time : {:.2f}s'.format(epoch, nepochs, train_loss / ntrain, test_loss / ntest, time.time() - t))
	torch.save(autoencoder.encoder.state_dict(), os.path.join(encoder_folder, dataset_name + '-encoder' + name[2:] + '.pth'))
	torch.save(autoencoder.decoder.state_dict(), os.path.join(decoder_folder, dataset_name + '-decoder' + name[2:] + '.pth'))

def train_and_save_all(filters_list, ds_list, bn_list, penalty_list, lmbda_list, nepochs = 20):
	for filters in filters_list:
		for ds in ds_list:
			for bn in bn_list:
				for penalty in penalty_list:
					if penalty == 0:
						train_and_save(filters, ds, bn, 0, 0, nepochs)
					else:
						for lmbda in lmbda_list:
							train_and_save(filters, ds, bn, penalty, lmbda, nepochs)

def create_model(filters, ds, bn):
	autoencoder = Autoencoder(data_shape[1], filters, ds, bn)
	print(autoencoder)
	summary(autoencoder, data_shape[1:])

def create_all(filters_list, ds_list, bn_list):
	for filters in filters_list:
		for ds in ds_list:
			for bn in bn_list:
				create_model(filters, ds, bn)
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-da", "--dataset", default = 'mnist', choices = ['mnist', 'cifar10', 'cifar100'])
	parser.add_argument("-fl", "--filters", type = int, default = [16, 32, 64, 100, 150], nargs = '*')
	parser.add_argument("-ds", "--downsample", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-bn", "--batchnorm", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-pe", "--penalty", type = int, default = [0], choices = [0, 1 , 2], nargs = '*')
	parser.add_argument("-la", "--lmbda", type = float, default = [0.1, 0.01, 0.001, 0.0001], nargs = '*')
	parser.add_argument("-ne", "--nepochs", type = int, default = 20)
	args = parser.parse_args()

	batch_size = 128
	dataset_name = args.dataset
	trainloader, testloader, data_shape, n_classes, mean, std = dataloaders(dataset_name, batch_size)
	ntrain, ntest = len(trainloader), len(testloader)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	encoder_folder = dataset_name + '-encoder2-weights-temp'
	decoder_folder = dataset_name + '-decoder2-weights-temp'
	examples_folder = dataset_name + '-autoencoder2-examples-temp'
	make_folder(encoder_folder)
	make_folder(decoder_folder)
	make_folder(examples_folder)

	print(dataset_name, data_shape, args.filters, args.downsample, args.batchnorm, args.penalty, args.lmbda, args.nepochs)
	# train_and_save_all(args.filters, args.downsample, args.batchnorm, args.penalty, args.lmbda, args.nepochs)
	create_all(args.filters, args.downsample, args.batchnorm)

	



