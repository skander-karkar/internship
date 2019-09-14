# last, best
import numpy as np, time, sys, argparse, os
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type = int, choices = [0, 1, 2, 3], default = 1)
parser.add_argument("-m", "--model", type = int, choices = [1, 2, 3], default = 1)
parser.add_argument("-e", "--epochs", type = int, default = 30)
parser.add_argument("-c", "--clip", type = int, default = 5)
parser.add_argument("-r", "--random", type = int, choices = [0, 1], default = 0)
args = parser.parse_args()

DATASET = args.data # 0 FAKE 1 MNIST 2 CIFAR10 3 CIFAR100
DATASET_NAMES = ['FAKE', 'MNIST', 'CIFAR10', 'CIFAR100']
INPUT_SHAPES = [(1, 3, 32, 32), (1, 1, 28, 28), (1, 3, 32, 32), (1, 3, 32, 32)]
DATASET_CLASSES = [10, 10, 10, 100]
DATASET_NAME = DATASET_NAMES[DATASET]
INPUT_SHAPE = INPUT_SHAPES[DATASET]
N_CLASSES = DATASET_CLASSES[DATASET]

MEANS = [(0.49139968, 0.48215841, 0.44653091), (0.1306604762738429, ), (0.5, 0.5, 0.5)]
STDS = [(0.24703223, 0.24348513, 0.26158784), (0.30810780717887876, ), (0.5, 0.5, 0.5)]

MODEL = args.model - 1 # 1 ResNet 2 OneRepResNet 3 AvgPoolResNet
MODEL_NAMES = ['ResNet', 'OneRepResNet', 'AvgPoolResNet']
MODEL_STAGES = [3, 1, 3]
MODEL_BLOCKS = [18, 10, 10]
MODEL_FILTERS = [(16, 32, 64), 100, 150]
MODEL_NAME = MODEL_NAMES[MODEL]
N_STAGES = MODEL_STAGES[MODEL]
N_BLOCKS = MODEL_BLOCKS[MODEL]

BATCH_SIZE = 128 
N_EPOCHS = args.epochs
CLIP = args.clip

FILENAME = MODEL_NAME + 'File1' + DATASET_NAME + '-' + time.strftime("%Y%m%d-%H%M%S")
FOLDER = os.path.join(os.getcwd(), 'iterative_inference_results')
if not os.path.exists(FOLDER):
	os.makedirs(FOLDER)
sys.stdout = open(os.path.join(FOLDER, 'Output' + FILENAME + '.txt'), 'wt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if not args.random:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(1)
	np.random.seed(1)

def cifar10_dataloaders():
	transform = [transforms.ToTensor(), transforms.Normalize(MEANS[0], STDS[0])]
	data_aug = [transforms.RandomCrop(INPUT_SHAPE[-1], padding = 4), transforms.RandomHorizontalFlip()]
	trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.Compose(data_aug + transform))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
	testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
	return trainloader, testloader

def cifar100_dataloaders():
	transform = [transforms.ToTensor(), transforms.Normalize(MEANS[0], STDS[0])]
	data_aug = [transforms.RandomCrop(INPUT_SHAPE[-1], padding = 4), transforms.RandomHorizontalFlip()]
	trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = transforms.Compose(data_aug + transform))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
	testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
	return trainloader, testloader

def mnist_dataloaders():
	transform = [transforms.ToTensor(), transforms.Normalize(MEANS[1], STDS[1])]
	trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.Compose(transform))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
	testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
	return trainloader, testloader

def fake_dataloaders():
	transform = [transforms.ToTensor(), transforms.Normalize(MEANS[2], STDS[2])]
	trainset = torchvision.datasets.FakeData(size = 400, image_size = INPUT_SHAPE[1:], num_classes = N_CLASSES, transform = transforms.Compose(transform))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
	testset = torchvision.datasets.FakeData(size = 200, image_size = INPUT_SHAPE[1:], num_classes = N_CLASSES, transform = transforms.Compose(transform))
	testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
	return trainloader, testloader
    
def initialize_weights(module):
	if isinstance(module, nn.Conv2d):
		nn.init.kaiming_normal_(module.weight.data, mode = 'fan_out', nonlinearity = 'relu')
	elif isinstance(module, nn.BatchNorm2d):
		nn.init.constant_(module.weight, 1)
		nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.Linear):
		nn.init.kaiming_normal_(module.weight)
		nn.init.constant_(module.bias, 0)

class ResBlock(nn.Module):
	def __init__(self, filters):
		super(ResBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(filters, track_running_stats = True)
		self.cv1 = nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(filters, track_running_stats = True)
		self.cv2 = nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False)
	def forward(self, x):
		z = F.relu(self.bn1(x))
		z = self.cv1(z)
		z = F.relu(self.bn2(z))
		z = self.cv2(z)
		z = z + x
		return z

class FirstResBlock(nn.Module):
	def __init__(self, filters):
		super(FirstResBlock, self).__init__()
		self.cv1 = nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(filters, track_running_stats = True)
		self.cv2 = nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False)
	def forward(self, x):
		z = self.cv1(x)
		z = F.relu(self.bn2(z))
		z = self.cv2(z)
		z = z + x
		return z

def make_stage(n_blocks, filters, first = False):
	stage = nn.Sequential()
	for i in range(n_blocks):
		name = 'block{}'.format(i + 1)
		if first and i == 0 :
			stage.add_module(name, FirstResBlock(filters))
		else:
			stage.add_module(name, ResBlock(filters))
	return stage

class ResNet(nn.Module):
	def __init__(self, n_blocks = MODEL_BLOCKS[0], filters = MODEL_FILTERS[0]):
		super(ResNet, self).__init__()
		self.cvIn = nn.Conv2d(INPUT_SHAPE[1], filters[0], kernel_size = 3, padding = 1, bias = False)
		self.bnIn = nn.BatchNorm2d(filters[0], track_running_stats = True)
		self.rlIn = nn.ReLU()
		self.stage1 = make_stage(n_blocks, filters[0], True)
		self.cv1 = nn.Conv2d(filters[0], filters[1], kernel_size = 1, stride = 2, padding = 0, bias = False)
		self.stage2 = make_stage(n_blocks, filters[1])
		self.cv2 = nn.Conv2d(filters[1], filters[2], kernel_size = 1, stride = 2, padding = 0, bias = False)
		self.stage3 = make_stage(n_blocks, filters[2])
		self.bnOut = nn.BatchNorm2d(filters[2], track_running_stats = True)
		self.avgpoolOut = nn.AvgPool2d(8, 8)
		with torch.no_grad():
			self.feature_size = self.forward_conv(torch.zeros(*INPUT_SHAPE)).view(-1).shape[0]
		self.fc = nn.Linear(self.feature_size, N_CLASSES)
		self.apply(initialize_weights)
	def forward_conv(self, x):
		x = self.cvIn(x)
		x = self.bnIn(x)
		x = self.rlIn(x)
		x = self.stage1(x)
		x = self.cv1(x)
		x = self.stage2(x)
		x = self.cv2(x)
		x = self.stage3(x)
		x = F.relu(self.bnOut(x))
		x = self.avgpoolOut(x)
		return x
	def forward(self, x):
		x = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class OneRepResNet(nn.Module):
	def __init__(self, n_blocks = MODEL_BLOCKS[1], filters = MODEL_FILTERS[1]):
		super(OneRepResNet, self).__init__()
		self.cvIn = nn.Conv2d(INPUT_SHAPE[1], filters, kernel_size = 3, padding = 1, bias = False)
		self.bnIn = nn.BatchNorm2d(filters, track_running_stats = True)
		self.rlIn = nn.ReLU()
		self.stage1 = make_stage(n_blocks, filters, True)
		self.bnOut = nn.BatchNorm2d(filters, track_running_stats = True)
		self.avgpoolOut = nn.AvgPool2d(8, 8)
		with torch.no_grad():
			self.feature_size = self.forward_conv(torch.zeros(*INPUT_SHAPE)).view(-1).shape[0]
		self.fc = nn.Linear(self.feature_size, N_CLASSES)
		self.apply(initialize_weights)
	def forward_conv(self, x):
		x = self.cvIn(x)
		x = self.bnIn(x)
		x = self.rlIn(x)
		x = self.stage1(x)
		x = F.relu(self.bnOut(x))
		x = self.avgpoolOut(x)
		return x
	def forward(self, x):
		x = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class AvgPoolResNet(nn.Module):
	def __init__(self, n_blocks = MODEL_BLOCKS[2], filters = MODEL_FILTERS[2]):
		super(AvgPoolResNet, self).__init__()
		self.cvIn = nn.Conv2d(INPUT_SHAPE[1], filters, kernel_size = 3, padding = 1, bias = False)
		self.bnIn = nn.BatchNorm2d(filters, track_running_stats = True)
		self.rlIn = nn.ReLU()
		self.stage1 = make_stage(n_blocks, filters, True)
		self.avgpool1 = nn.AvgPool2d(2, 2)
		self.stage2 = make_stage(n_blocks, filters)
		self.avgpool2 = nn.AvgPool2d(2, 2)
		self.stage3 = make_stage(n_blocks, filters)
		self.avgpool3 = nn.AvgPool2d(2, 2)
		self.bnOut = nn.BatchNorm2d(filters, track_running_stats = True)
		self.avgpoolOut = nn.AvgPool2d(8, 8)
		with torch.no_grad():
			self.feature_size = self.forward_conv(torch.zeros(*INPUT_SHAPE)).view(-1).shape[0]
		self.fc = nn.Linear(self.feature_size, N_CLASSES)
		self.apply(initialize_weights)
	def forward_conv(self, x):
		x = self.cvIn(x)
		x = self.bnIn(x)
		x = self.rlIn(x)
		x = self.stage1(x)
		x = self.avgpool1(x)
		x = self.stage2(x)
		x = self.avgpool2(x)
		x = self.stage3(x)
		x = self.avgpool3(x)
		x = F.relu(self.bnOut(x))
		x = self.avgpoolOut(x)
		return x
	def forward(self, x):
		x = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class AverageMeter(object):
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, num):
		self.val = val
		self.sum += val * num
		self.count += num
		self.avg = self.sum / self.count

def update_meters(y, pred, loss, loss_meter, accuracy_meter, t = None, time_meter = None):
	num = len(y)
	correct = (pred == y).sum().item()
	accuracy = correct / num
	loss_meter.update(loss, num)
	accuracy_meter.update(accuracy, num)
	if t is not None and time_meter is not None :
		time_meter.update(t, 1)

def train_one_epoch(e, model, optimizer, criterion, trainloader, train_loss, train_acc):
	t0 = time.time()
	model.train()
	n_batches = len(trainloader)
	loss_meter, accuracy_meter, time_meter = AverageMeter(), AverageMeter(), AverageMeter()
	for i, (x, y) in enumerate(trainloader):
		x, y = x.to(device), y.to(device)
		t1 = time.time()
		optimizer.zero_grad()
		out = model(x)
		loss = criterion(out, y)
		loss.backward()
		if CLIP > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
		optimizer.step()
		_, pred = torch.max(out.data, 1)
		update_meters(y, pred, loss.item(), loss_meter, accuracy_meter, time.time() - t1, time_meter)
		if i % 100 == 0 :
			metrics = (e + 1, N_EPOCHS, i + 1, n_batches, loss_meter.avg, accuracy_meter.avg, time_meter.avg)
			print('Epoch {}/{} Batch {}/{} Train loss {:.4f} Train acc {:.4f} Avg batch time {:.4f}s'.format(*metrics))
	train_loss.append(loss_meter.avg)
	train_acc.append(accuracy_meter.avg)
	return time.time() - t0

def test(model, criterion, testloader, val_loss, val_acc):
	model.eval()
	loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
	for (x, y) in testloader:
		x, y = x.to(device), y.to(device)
		with torch.no_grad():
			out = model(x)
			loss = criterion(out, y)
			_, pred = torch.max(out.data, 1)
		update_meters(y, pred, loss.item(), loss_meter, accuracy_meter)
	val_loss.append(loss_meter.avg)
	val_acc.append(accuracy_meter.avg)
	return accuracy_meter.avg

def simple_run(model, criterion, dataloader):
	model.eval()
	for (x, y) in dataloader:
		x, y = x.to(device), y.to(device)
		out = model(x)
		loss = criterion(out, y)
		loss.backward()

def train(model, optimizer, scheduler, criterion, trainloader, testloader):
	model.train()
	train_loss, train_acc, val_loss, val_acc = [], [], [], []
	print('-' * 9, 'Begin training')
	t0 = time.time()
	for e in range(N_EPOCHS):
		print('Epoch {}/{}'.format(e + 1, N_EPOCHS))
		epoch_time = train_one_epoch(e, model, optimizer, criterion, trainloader, train_loss, train_acc)
		val_acc_ = test(model, criterion, testloader, val_loss, val_acc)
		scheduler.step()
		total_time = time.time() - t0
		print('Epoch {}/{} Val acc {:.4f} Epoch train time {:.4f}s Total time {:.4f}s'.format(e + 1, N_EPOCHS, val_acc_, epoch_time, total_time))
	return train_loss, train_acc, val_loss, val_acc

def input_hook(name, f, fn, ratio_meters, cosine_meters, mod, inp, out):
	j, i = int(name[5]), int(name[12:]) 
	h_ = inp[0] if type(inp) is tuple else inp
	hn_ = l2norm(h_.cpu().detach().numpy().copy())
	fn_ = fn.get(name + '.cv2')
	ratio = fn_ / np.where(hn_ == 0, 0.0001, hn_)
	ratio_meters[(j,i)].update(np.mean(ratio), ratio.shape[0])

def output_hook(name, f, fn, ratio_meters, cosine_meters, mod, inp, out):
	f_ = out[0] if type(out) is tuple else out
	f_ = f_.cpu().detach().numpy().copy()
	fn_ = l2norm(f_)
	f[name], fn[name] = f_, fn_

def outgrad_hook(name, f, fn, ratio_meters, cosine_meters, mod, ginp, gout):
	g_ = gout[0] if type(gout) is tuple and gout[0] is not None else gout
	g_ = g_.cpu().detach().numpy().copy()
	gn_ = l2norm(g_)
	if name == 'rlIn':
		j, i = 1, 1
	elif name in ['cv1', 'avgpool1']:
		j, i = 2, 1
	elif name in ['cv2', 'avgpool2']:
		j, i = 3, 1
	else :
		j, i = int(name[5]), int(name[12:]) + 1
	s = 'stage' + str(j) + '.' + 'block' + str(i) + '.cv2'
	f_, fn_ = f[s], fn[s]
	cos = np.sum(np.multiply(f_, g_), axis = (1, 2, 3))
	cos = cos / np.multiply(np.where(fn_ == 0, 0.0001, fn_), np.where(gn_ == 0, 0.0001, gn_))
	cosine_meters[(j, i)].update(np.mean(cos), cos.shape[0])
	
def l2norm(x):
	return np.sqrt(np.sum(x ** 2, axis = (1, 2, 3)))

def register_hooks(model, input_hook, output_hook, outgrad_hook, *args):
	for name, m in model.named_modules():
		if name in ['rlIn', 'cv1', 'cv2', 'avgpool1', 'avgpool2'] :
			m.register_backward_hook(partial(outgrad_hook, name, *args))
		elif len(name) in [13, 14] and name[0:5] == 'stage' and name[7:12] == 'block' :
			m.register_forward_hook(partial(input_hook, name, *args))
			if int(name[12:]) < N_BLOCKS:
				m.register_backward_hook(partial(outgrad_hook, name, *args))
		elif len(name) in [17, 18] and name[0:5] == 'stage' and name[7:12] == 'block' and name[-3:] == 'cv2' :
			m.register_forward_hook(partial(output_hook, name, *args))

def apply_hooks(model, criterion, testloader):
	f, fn = dict(), dict()
	rat_meters = {(j, i) : AverageMeter() for j in range(1, N_STAGES + 1) for i in range(1, N_BLOCKS + 1)}
	cos_meters = {(j, i) : AverageMeter() for j in range(1, N_STAGES + 1) for i in range(1, N_BLOCKS + 1)}
	register_hooks(model, input_hook, output_hook, outgrad_hook, f, fn, rat_meters, cos_meters)
	simple_run(model, criterion, testloader)
	r = [rat_meters[(j, i)].avg for j in range(1, N_STAGES + 1) for i in range(1, N_BLOCKS + 1)]
	c = [cos_meters[(j, i)].avg for j in range(1, N_STAGES + 1) for i in range(1, N_BLOCKS + 1)]
	return r, c

def set_lr(optimizer, lr):
	for g in optimizer.param_groups:
		g['lr'] = lr

def main():
	print(FILENAME)
	dataloaders = [fake_dataloaders, mnist_dataloaders, cifar10_dataloaders, cifar100_dataloaders]
	trainloader, testloader = dataloaders[DATASET]()
	models = [ResNet(), OneRepResNet(), AvgPoolResNet()]
	model = models[MODEL]
	model.to(device)
	print(model)
	summary(model, INPUT_SHAPE[1:])
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0001)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [120, 160, 200], gamma = 0.2)
	train_loss, train_acc, val_loss, val_acc = train(model, optimizer, scheduler, criterion, trainloader, testloader)
	print('-' * 9, 'train loss\n', train_loss, '\n', '-' * 9, 'train_acc\n', train_acc, '\n', '-' * 9, 'val loss\n', val_loss, '\n', '-' * 9, 'val acc\n', val_acc)
	torch.save(model.state_dict(), os.path.join(FOLDER, 'Model' + FILENAME + '.pth'))
	ratios, cosines = apply_hooks(model, criterion, testloader)
	print('-' * 9, 'ratios\n', ratios, '\n', '-' * 9, 'cosines\n', cosines)

if __name__ == '__main__':
    main()

		










