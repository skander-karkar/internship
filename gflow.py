import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders2 import dataloaders
from utils3 import *
from torchsummary import summary
import time, numpy as np, matplotlib.pyplot as plt, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial
from sklearn.manifold import TSNE

def pretrained_ae_files(filters, ds, bn, pen, lmbda, folder):
	name = ('ds' if ds else '') + ('bn' if bn else '') + str(filters) + ('pen' + str(pen) if pen else '') + ('lmbda' + str(lmbda) if pen else '') + '.pth'
	encoder_pre, decoder_pre = dataset_name + '-encoder2-', dataset_name + '-decoder2-'
	encoder_file = os.path.join(os.getcwd(), 'autoencoders2', encoder_pre + 'weights-all', encoder_pre + 'weights' + str(folder), encoder_pre + name)
	decoder_file = os.path.join(os.getcwd(), 'autoencoders2', decoder_pre + 'weights-all', decoder_pre + 'weights' + str(folder), decoder_pre + name)
	return encoder_file, decoder_file

def load_autoencoder(in_channels, filters, ds, bn, enc_file, dec_file):
	encoder, decoder = create_autoencoder(in_channels, filters, ds, bn)
	encoder.load_state_dict(torch.load(enc_file))
	decoder.load_state_dict(torch.load(dec_file))
	encoder.eval()
	decoder.eval()
	return encoder, decoder

def test_autoencoder(datashape, filters, ds, encoder, decoder, testloader, mean, std):
	encoder.to(device)
	print('-' * 64, 'encoder', encoder)
	summary(encoder, datashape[1:])
	decoder.to(device)
	print('-' * 64, 'decoder', decoder)
	criterion = nn.MSELoss()
	test_loss, idx_batch = 0, 4
	for i, (x, _) in enumerate(testloader):
		x = x.to(device)
		z = encoder(x)
		y = decoder(z)
		loss = criterion(y, x)
		test_loss += loss.item()
		if i == idx_batch:
			idx_images = np.random.choice(x.size()[0], 5, replace = False)
			x_ = x.cpu().detach().numpy().copy()[idx_images, :, :, :]
			y_ = y.cpu().detach().numpy().copy()[idx_images, :, :, :]
			show_autoencoder_images(x_, y_, mean, std, 'test-ae2.png')
			break
	test_loss /= (i + 1)
	print('Test loss : {:.4f}'.format(test_loss))

class FirstResBlock(nn.Module):
	def __init__(self, filters, batchnorm, bias, h = 1):
		super(FirstResBlock, self).__init__()
		self.h = h
		self.batchnorm = batchnorm
		self.cv1 = nn.Conv2d(filters, filters, 3, 1, 1, bias = bias)
		if self.batchnorm:
			self.bn2 = nn.BatchNorm2d(filters)
		self.cv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = self.cv1(x)
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.h * z, z 

class ResBlock(nn.Module):
	def __init__(self, filters, batchnorm, bias, h = 1):
		super(ResBlock, self).__init__()
		self.h = h
		self.batchnorm = batchnorm
		if self.batchnorm :
			self.bn1 = nn.BatchNorm2d(filters)
		self.cv1 = nn.Conv2d(filters, filters, 3, 1, 1, bias = bias)
		if self.batchnorm :
			self.bn2 = nn.BatchNorm2d(filters)
		self.cv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = functional.relu(self.bn1(x)) if self.batchnorm else functional.relu(x)
		z = self.cv1(z) 
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self. h * z, z

class OneRepResNet(nn.Module):
	def __init__(self, datashape, nclasses, filters, nblocks, batchnorm, bias, encoder, initname = 'orthogonal', initgain = 0.01,
				 learn_cl = True, cl_name = '3Lin', cl_file = None):
		super(OneRepResNet, self).__init__()
		self.classifier_name = cl_name
		self.encoder = encoder
		for param in self.encoder.parameters():
			param.requires_grad = False
		h = 1 / nblocks
		self.stage1 = nn.ModuleList([FirstResBlock(filters, batchnorm, bias, h) if i == 0 else ResBlock(filters, batchnorm, bias, h) for i in range(nblocks)])
		feature_shape = list(self.encoder(torch.ones(*datashape).to(device)).shape)
		initialization = partial(initialize, initname, initgain)
		self.stage1.apply(initialization)
		if learn_cl:
			self.classifier = create_classifier(cl_name, nclasses, feature_shape, filters)
			self.classifier.apply(initialization)
		else:
			self.classifier = load_classifier(cl_name, filters, feature_shape, cl_file)
			for param in self.classifier.parameters():
				param.requires_grad = False
	def forward(self, x):
		x = self.encoder(x)
		rs = []
		for block in self.stage1:
			x, r = block(x)
			rs.append(r)
		if self.classifier_name[-3:] == 'Lin':
			x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_name, batchsize, filters = 'cifara10', 128, 100
trainloader, valloader, testloader, datashape, nclasses, mean, std = dataloaders(dataset_name, batchsize)

ae_penalty, ae_lambda, ae_folder = 0, 0, 3
(ae_ds, ae_bn) = (False, True) if dataset_name == 'cifar100' else (True, False)
encoder_file, decoder_file = pretrained_ae_files(filters, ae_ds, ae_bn, ae_penalty, ae_lambda, ae_folder)
encoder, decoder = load_autoencoder(datashape[1], filters, ae_ds, ae_bn, encoder_file, decoder_file)
encoder.to(device)
decoder.to(device)

nmodels, nblocks, batchnorm, initname, initgain = 9, 9, 2, 3, 1, 'orthogonal', 0.01
models = [OneRepResNet(datashape, nclasses, filters, nblocks, batchnorm, 0, encoder) for _ in range(nmodels)]
for model in models:
	model.to(device)

learningrate, beta1, beta2, nepochs, tau = 0.01, 0.9, 0.99, 10, 0.1
optimizers = [optim.Adam(model.parameters(), lr = learningrate, betas = (beta1, beta2)) for model in models]
criterion = nn.CrossEntropyLoss()

for e in range(nepochs):
	running_loss = 0
	for j, (x, y) in enumerate(trainloader):
		x, y = x.to(device), y.to(device)
		z = Variable(x, requires_grad = False)
		for i, model in enumerate(models):
			optimizers[i].zero_grad()
			out, rs = model(z)
			n = z.shape[0]
			transport = sum([torch.sum(r ** 2) for r in rs]) / (n * nblocks)
			target = criterion(out, y)
			loss = target + transport / (2 * tau)
			loss.backward()
			optimizers[i].step()
			z = Variable(out, requires_grad = False)
			if i == nblocks - 1:
				running_loss += target.item()
	print('epoch', e + 1, 'loss', running_loss)

for model in models:
	model.eval()
