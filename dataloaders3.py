# valation set
import torch, torch.utils.data as torchdata
import torchvision, torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def cifar10_dataloaders(batchsize, trainsize = None, split = 0.5):
	datashape, mean, std, nclasses = (1, 3, 32, 32), (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784), 10
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.RandomCrop(datashape[-1], padding = 4), transforms.RandomHorizontalFlip()]
	trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.Compose(data_aug + transform))
	testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	trainloader = get_subset_loader(trainset, batchsize, trainsize)
	valloader, testloader = get_split_loaders(testset, batchsize, split)
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def cifar100_dataloaders(batchsize, trainsize = None, split = 0.5):
	datashape, mean, std, nclasses = (1, 3, 32, 32), (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784), 100
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.RandomCrop(datashape[-1], padding = 4), transforms.RandomHorizontalFlip()]
	trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = transforms.Compose(data_aug + transform))
	testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	trainloader = get_subset_loader(trainset, batchsize, trainsize)
	valloader, testloader = get_split_loaders(testset, batchsize, split)
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def mnist_dataloaders(batchsize, trainsize = None, split = 0.5):
	datashape, mean, std, nclasses = (1, 1, 28, 28), (0.1306604762738429, ), (0.30810780717887876, ), 10
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.Compose(transform))
	testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	trainloader = get_subset_loader(trainset, batchsize, trainsize)
	valloader, testloader = get_split_loaders(testset, batchsize, split)
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def imagenet_dataloaders(batchsize, trainsize = 1, testsize = 0.01):
	datashape, mean, std, nclasses = (1, 3, 224, 224), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 1000
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
	test_trs = [transforms.Resize(256), transforms.CenterCrop(224)]
	trainset = torchvision.datasets.ImageNet(root = './data', split = 'train', download = True, transform = transforms.Compose(data_aug + transform))
	if trainsize == 1 :
		trainloader, testloader = get_split_loaders(trainset, batchsize, testsize)
	else :
		trainloader, testloader = get_subset_split_loaders(trainset, batchsize, trainsize, testsize)
	return trainloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def get_subset_loader(dataset, batchsize, size):
	(sampler, shuffle) = (None, True) if size in [None, 'all', 0, 1] else (SubsetRandomSampler(np.random.choice(range(len(dataset)), size, False)), False)
	return torchdata.DataLoader(dataset, batch_size = batchsize, shuffle = shuffle, num_workers = 2, sampler = sampler)

def get_subset_split_loaders(dataset, batchsize, trainsize = 0.7, testsize = 0.01):
	n = len(dataset)
	indices = list(range(n))
	np.random.shuffle(indices)
	strain = int(np.floor(trainsize * n))
	stest = int(np.floor(testsize * n))
	idxtrain, idxtest = indices[: strain], indices[strain: strain + stest]
	trainsampler, testsampler = SubsetRandomSampler(idxtrain), SubsetRandomSampler(idxtest)
	trainloader = torchdata.DataLoader(dataset, batch_size = batchsize, shuffle = False, num_workers = 2, sampler = trainsampler)
	testloader = torchdata.DataLoader(dataset, batch_size = batchsize, shuffle = False, num_workers = 2, sampler = testsampler)
	return trainloader, testloader

def get_split_loaders(dataset, batchsize, split = 0.5):
	n = len(dataset)
	indices = list(range(n))
	np.random.shuffle(indices)
	sp = int(np.floor(split * n))
	idx1, idx2 = indices[sp:], indices[:sp]
	sampler1, sampler2 = SubsetRandomSampler(idx1), SubsetRandomSampler(idx2)
	loader1 = torchdata.DataLoader(dataset, batch_size = batchsize, shuffle = False, num_workers = 2, sampler = sampler1)
	loader2 = torchdata.DataLoader(dataset, batch_size = batchsize, shuffle = False, num_workers = 2, sampler = sampler2)
	return loader1, loader2

def fake_dataloaders(batchsize, datashape, nclasses):
	transform = [transforms.ToTensor()]
	trainset = torchvision.datasets.FakeData(size = 400, image_size = datashape[1:], num_classes = nclasses, transform = transforms.Compose(transform))
	trainloader = torchdata.DataLoader(trainset, batch_size = batchsize, shuffle = True, num_workers = 2)
	valset = torchvision.datasets.FakeData(size = 200, image_size = datashape[1:], num_classes = nclasses, transform = transforms.Compose(transform))
	valloader = torchdata.DataLoader(valset, batch_size = batchsize, shuffle = True, num_workers = 2)
	testset = torchvision.datasets.FakeData(size = 200, image_size = datashape[1:], num_classes = nclasses, transform = transforms.Compose(transform))
	testloader = torchdata.DataLoader(testset, batch_size = batchsize, shuffle = True, num_workers = 2)
	return trainloader, valloader, testloader, datashape, nclasses, None, None

def dataloaders(name, batchsize, trainsize = None, split = 0.5):
	if name == 'imagenet':
		return imagenet_dataloaders(batchsize, trainsize, split)
	if name == 'mnist':
		return mnist_dataloaders(batchsize, trainsize, split)
	if name == 'cifar10':
		return cifar10_dataloaders(batchsize, trainsize, split)
	if name == 'cifar100':
		return cifar100_dataloaders(batchsize, trainsize, split)
	if name == 'fake_like_mnist':
		return fake_dataloaders(batchsize, (1, 1, 28, 28), 10)
	if name == 'fake_like_cifar10':
		return fake_dataloaders(batchsize, (1, 3, 32, 32), 10)
	if name == 'fake_like_cifar100':
		return fake_dataloaders(batchsize, (1, 3, 32, 32), 100)
	else:
		raise ValueError('unknown dataset: ' + name)




