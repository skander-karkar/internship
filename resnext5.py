# resnext
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders3 import dataloaders
from utils4 import *
from torchsummary import summary
import time, math, numpy as np, matplotlib.pyplot as plt, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial

class ResNextBlock(nn.Module):
	def __init__(self, infilters = 256, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, stride = 1, relu = True, downsample = None):
		super(ResNextBlock, self).__init__()
		self.relu = relu 
		self.intfilters = cardinality * math.floor(planes * width / base)
		self.outfilters = planes * expansion
		self.cv1 = nn.Conv2d(infilters, self.intfilters, 1, 1, 0, bias = False)
		self.bn1 = nn.BatchNorm2d(self.intfilters)
		self.cv2 = nn.Conv2d(self.intfilters, self.intfilters, 3, stride, 1, groups = cardinality, bias = False)
		self.bn2 = nn.BatchNorm2d(self.intfilters)
		self.cv3 = nn.Conv2d(self.intfilters, self.outfilters, 1, 1, 0, bias = False)
		self.bn3 = nn.BatchNorm2d(self.outfilters)
		self.downsample = downsample
	def forward(self, x):
		r = functional.relu(self.bn1(self.cv1(x)), inplace = True)
		r = functional.relu(self.bn2(self.cv2(r)), inplace = True)
		r = functional.relu(self.bn3(self.cv3(r)), inplace = True) if self.relu else self.bn3(self.cv3(r))
		if self.downsample is not None:
			x = self.downsample(x)
		x = x + r if not self.relu else functional.relu(x + r, inplace = True)
		return x, r

class ResNextStage(nn.Module):
	def __init__(self, nb, inf = 256, pln = 64, exp = 4, card = 32, width = 4, base = 64, stride = 1, rel = True):
		super(ResNextStage, self).__init__()
		intf = pln * exp
		ds = nn.Sequential(nn.Conv2d(inf, intf, 1, stride, bias = False), nn.BatchNorm2d(intf)) if stride != 1 or inf != intf else None
		bl = [ResNextBlock(inf, pln, exp, card, width, base, stride, rel, ds) if i == 0 else ResNextBlock(intf, pln, exp, card, width, base, 1, rel) for i in range(nb)]
		self.blocks = nn.ModuleList(bl)
	def forward(self, x):
		residus = []
		for block in self.blocks :
			x, r = block(x)
			residus.append(r)
		return x, residus

class ResNext(nn.Module):
	def __init__(self, datashape, nclasses, nblocks = [3, 4, 6, 3], infilters = 64, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, relu = True):
		super(ResNext, self).__init__()
		self.cv = nn.Conv2d(3, infilters, 7, 2, 3, bias = False)
		self.bn = nn.BatchNorm2d(infilters)
		self.maxpool = nn.MaxPool2d(3, 2, 1)
		self.stage1 = ResNextStage(nblocks[0], infilters * 1, planes * 1, expansion, cardinality, width, base, 1, relu)
		self.stage2 = ResNextStage(nblocks[1], infilters * 4, planes * 2, expansion, cardinality, width, base, 2, relu)
		self.stage3 = ResNextStage(nblocks[2], infilters * 8, planes * 4, expansion, cardinality, width, base, 2, relu)
		self.stage4 = ResNextStage(nblocks[3], infilters * 16, planes * 8, expansion, cardinality, width, base, 2, relu)
		self.avgpool = nn.AvgPool2d(7, 1)
		with torch.no_grad():
			self.feature_size = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
			print('feature size:', self.feature_size)
		self.fc = nn.Linear(self.feature_size, nclasses)
	def forward_conv(self, x):
		residus = dict()
		x = self.maxpool(functional.relu(self.bn(self.cv(x)), inplace = True))
		x, residus[1] = self.stage1(x)
		x, residus[2] = self.stage2(x)
		x, residus[3] = self.stage3(x)
		x, residus[4] = self.stage4(x)
		x = self.avgpool(x)
		return x, residus
	def forward(self, x):
		x, residus = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x, residus

def train(model, optimizer, scheduler, criterion, trainloader, testloader, lmt, lml0, tau, uzs, nepochs = 300, clip = 0):
	train_loss, train_acc1, train_acc5, test_loss, test_acc1, test_acc5, lml, t0, it = [], [], [], [], [], [], lml0, time.time(), 0
	print('--- Begin trainning')
	for e in range(nepochs):
		model.train()
		t1, loss_meter, acc1_meter, acc5_meter, time_meter = time.time(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
		for j, (x, y) in enumerate(trainloader):
			t2, it = time.time(), it + 1
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			out, residus = model(x)
			loss = lml * criterion(out, y) + lmt * sum([sum([torch.mean(r ** 2) for r in residus[i]]) for i in range(1, 5)])
			loss.backward()
			if clip > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()
			num = len(y)
			prec1, prec5 = topkaccuracy(out.data, y.data, topk = (1, 5))
			loss_meter.update(loss.item(), num)
			acc1_meter.update(prec1, num)
			acc5_meter.update(prec5, num)
			time_meter.update(time.time() - t2, 1)
			if uzs > 0 and it % uzs == 0 and lml < 6:
				out, residus = model(x)
				lml += tau * criterion(out, y).item()
			if j % 500 == 0 :
				metrics = (e + 1, nepochs, j + 1, len(trainloader), lml, loss_meter.avg, acc1_meter.avg, acc5_meter.avg, time_meter.avg)
				print('Epoch {}/{} Batch {}/{} | Lambda loss {:.4f} Train loss {:.4f} Train top1acc {:.4f} Train top5acc {:.4f} Avg batch time {:.4f}s'.format(*metrics))
		train_loss.append(loss_meter.avg)
		train_acc1.append(acc1_meter.avg)
		train_acc5.append(acc5_meter.avg)
		optimizer.zero_grad()
		test(model, criterion, testloader, test_loss, test_acc1, test_acc5)
		metrics = (e + 1, nepochs, test_acc1[-1], test_acc5[-1], time.time() - t1, time.time() - t0)
		print('******************************* Epoch {}/{} over | Test top1acc {:.4f} Test top5acc {:.4f} Epoch train time {:.4f}s Total time {:.4f}s'.format(*metrics))
		scheduler.step()
	return train_loss, test_acc1, test_acc5

def test(model, criterion, loader, test_loss, test_acc1, test_acc5):
	model.eval()
	loss_meter, acc1_meter, acc5_meter = AverageMeter(), AverageMeter(), AverageMeter()
	for j, (x, y) in enumerate(loader):
		with torch.no_grad():
			x, y = x.to(device), y.to(device)
			out, residus = model(x)
			ent = criterion(out, y)
			trs = sum([sum([torch.mean(r ** 2) for r in residus[i]]) for i in range(1, 5)])
			num = len(y)
			prec1, prec5 = topkaccuracy(out.data, y.data, topk = (1, 5))
			loss_meter.update(ent.item(), num)
			acc1_meter.update(prec1, num)
			acc5_meter.update(prec5, num)
	test_loss.append(loss_meter.avg)
	test_acc1.append(acc1_meter.avg)
	test_acc5.append(acc5_meter.avg)
	return 

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-dat", "--dataset", default = 'imagenet', choices = ['mnist', 'cifar10', 'cifar100', 'imagenet'])
	parser.add_argument("-bas", "--batchsize", type = int, default = 128)
	parser.add_argument("-lrr", "--learningrate", type = float, default = 0.1)
	parser.add_argument("-lmt", "--lambdatransport", type = float, default = 0)
	parser.add_argument("-lml", "--lambdaloss0", type = float, default = 1)
	parser.add_argument("-tau", "--tau", type = float, default = 0)
	parser.add_argument("-uzs", "--uzawasteps", type = int, default = 0)
	parser.add_argument("-clp", "--clip", type = float, default = 0)
	parser.add_argument("-nep", "--nepochs", type = int, default = 300)
	parser.add_argument("-inn", "--initname", default = 'kaiming', choices = ['orthogonal', 'normal', 'kaiming'])
	parser.add_argument("-ing", "--initgain", type = float, default = 0.01)
	parser.add_argument("-trs", "--trainsize", type = float, default = 1)
	parser.add_argument("-tss", "--testsize", type = int, default = 0.01)
	parser.add_argument("-see", "--seed", type = int, default = None)
	parser.add_argument("-rel", "--relu", type = int, default = 1, choices = [0, 1])
	args = parser.parse_args()

	if args.seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)

	print('--- resnext experiment from resnext5.py with parameters')
	print([(name, value) for name, value in vars(args).items()])

	trainloader, testloader, datashape, nclasses, mean, std = dataloaders(args.dataset, args.batchsize, args.trainsize, args.testsize)
	print(len(trainloader), len(testloader))
	model = ResNext(datashape, nclasses, relu = args.relu)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 1:
  		print('---', torch.cuda.device_count(), 'GPUs')
  		model = nn.DataParallel(model)
	model.to(device)
	print(model)
	# summary(model, datashape[1:])
	# model.apply(partial(initialize, args.initname, args.initgain))

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = args.learningrate, momentum = 0.9, weight_decay = 0.0001)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 225, 250], gamma = 0.1)

	tr_loss, ts_acc1, ts_acc5 = train(model, optimizer, scheduler, criterion, trainloader, testloader, args.lambdatransport, args.lambdaloss0, args.tau, args.uzawasteps)


