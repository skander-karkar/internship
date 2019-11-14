# uzawa, last
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
import pprint, inspect, collections, os, time, sys, argparse, ot, numpy as np, matplotlib.pyplot as plt, scipy.stats as st
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from itertools import product
from functools import partial
from matplotlib import colors

def makefolder(ndata, testsize, data, noise, factor, dataseed, modelseed, nblocks, datadim, hiddendim, batchnorm, nclasses, learnclassifier, 
			   yintercept, biginit, biginitstd, lambdatransport, lambdaloss0, tau, uzawasteps, batchsize, nepochs, learningrate, beta1, beta2):
	folder0 = ('circles' if data == 1 else 'moons') + '-dd' + str(datadim) + 'nc' + str(nclasses) 
	folder1 = 'points{}testsize{}'.format(ndata, testsize)
	folder2 = 'noise{}factor{}'.format(noise, factor) 
	folder3 = 'hiddendim' + str(hiddendim) + ('batchnorm' if batchnorm else '')
	folder4 = 'batchsize' + str(batchsize) + 'int' + str(yintercept) + 'lc' + str(learnclassifier) + ('bi' + str(biginitstd) if biginit else '')
	folder5 = 'blocks' + str(nblocks) 
	folder6 = 'uzawa-lambdaloss' + str(lambdaloss0) + 'tau' + str(tau) + 'us' + str(uzawasteps) if uzawasteps > 0 else 'lambdatransport' + str(lambdatransport)
	folder7 = 'ne{}lr{}b1{}b2{}'.format(nepochs, learningrate, beta1, beta2)
	folder8 = 'ds{}ms{}'.format(dataseed, modelseed)
	folder9 = time.strftime("%Y%m%d-%H%M%S")
	folder = os.path.join(os.getcwd(), 'figures3', folder0, folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8, folder9)
	os.makedirs(folder)
	return folder

def dataloaders(ndata, testsize, data, noise, factor, dataseed, batchsize):
	X, Y = make_circles(ndata, True, noise, dataseed, factor) if data == 1 else make_moons(ndata, True, noise, dataseed)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testsize, random_state = dataseed)
	X_train_ = torch.from_numpy(X_train).type(torch.FloatTensor) 
	Y_train_ = torch.from_numpy(Y_train).type(torch.LongTensor) 
	X_test_ = torch.from_numpy(X_test).type(torch.FloatTensor) 
	Y_test_ = torch.from_numpy(Y_test).type(torch.LongTensor) 
	train = torchdata.TensorDataset(X_train_, Y_train_)
	test = torchdata.TensorDataset(X_test_, Y_test_)
	trainloader = torchdata.DataLoader(train, batch_size = batchsize)
	testloader = torchdata.DataLoader(test, batch_size = batchsize)
	return trainloader, testloader, X, Y, X_test, Y_test

def plotdata(X, Y, learnclassifier, yintercept, title, folder):
	plt.figure(figsize = (15, 15))
	plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c = 'red')
	plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c = 'blue')
	if not learnclassifier:
		x = np.linspace(- 20, 10, 100)
		y = - x - yintercept
		plt.plot(x, y, '-g', label = 'linear classifier')
	plt.title(title)
	plt.savefig(os.path.join(folder, title + '.png'), bbox_inches = 'tight')
	plt.close()

def plotscores(losses, accuracy, folder):
	plt.figure(1)
	plt.subplot(211)
	plt.plot(losses)
	plt.ylabel('train loss')
	plt.subplot(212)
	plt.plot(accuracy)
	plt.xlabel('epoch')
	plt.ylabel('test accuracy')
	plt.savefig(os.path.join(folder, 'loss-acc.png'), bbox_inches = 'tight')
	plt.close()

def initialize_(biginit, biginitstd, module):
	if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
		nn.init.kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
	elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
		nn.init.constant_(module.weight, 1)
		nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.Linear):
		if biginit:
			nn.init.normal_(module.weight, mean = 0.0, std = biginitstd)
			nn.init.normal_(module.bias, mean = 0.0, std = biginitstd)
		else:
			nn.init.kaiming_normal_(module.weight)
			nn.init.constant_(module.bias, 0.0)

class ResBlock(nn.Module):
	def __init__(self, inputdim, hiddendim, batchnorm):
		super(ResBlock, self).__init__()
		self.batchnorm = batchnorm
		self.fc1 = nn.Linear(inputdim, hiddendim) 
		if batchnorm:
			self.bn = nn.BatchNorm1d(hiddendim, track_running_stats = True)
		self.fc2 = nn.Linear(hiddendim, inputdim)
	def forward(self, x):
		z = self.bn(self.fc1(x)) if self.batchnorm else self.fc1(x)
		z = functional.relu(z)
		z = self.fc2(z)
		return z + x, z

class OneRepResNet(nn.Module):
	def __init__(self, nblocks, inputdim, hiddendim, batchnorm, nclasses, learnclassifier, yintercept, initialize):
		super(OneRepResNet, self).__init__()
		self.blocks = nn.ModuleList([ResBlock(inputdim, hiddendim, batchnorm) for i in range(nblocks)])
		self.fcOut = nn.Linear(inputdim, nclasses)
		self.blocks.apply(initialize)
		if learnclassifier == 1:
			initialize(self.fcOut)
		else:
			with torch.no_grad():
				self.fcOut.weight = torch.nn.Parameter(torch.tensor([[1., 1.], [0., 0.]]))
				self.fcOut.bias = torch.nn.Parameter(torch.tensor([yintercept, 0.]))
			if learnclassifier == 0:
				for param in self.fcOut.parameters():
					param.requires_grad = False
	def forward(self, x):
		rs = []
		for block in self.blocks:
			x, r = block(x)
			rs.append(r)
		z = self.fcOut(x)
		return z, rs

def save_input_output_hook(name, mod, inp, out):
	global inps, outs
	inp0 = inp[0] if type(inp) is tuple else inp
	out0 = out[0] if type(out) is tuple else out
	inps[name].append(inp0.detach().numpy().copy())
	outs[name].append(out0.detach().numpy().copy())

def save_outgrad_hook(name, mod, ginp, gout):
	global gouts
	gout0 = gout[0] if type(gout) is tuple and gout[0] is not None else gout
	gouts[name].append(gout0.detach().numpy().copy())

def register_hooks(model):
	for name, m in model.named_modules():
		m.register_forward_hook(partial(save_input_output_hook, name))
		m.register_backward_hook(partial(save_outgrad_hook, name))

def W2(X1, X2):
	n = len(X1)
	C = np.zeros((n, n))
	for i in range(n):
		for j in range(n): 
			C[i, j] = np.linalg.norm(X1[i] - X2[j])
	optimal_plan = ot.emd([], [], C)
	optimal_cost = np.sum(optimal_plan * C)
	return optimal_cost

def train(model, nepochs, criterion, lambdatransport, lambdaloss0, tau, us, optimizer, trainloader, testloader, X_test, Y_test, nblocks, folder):
	ntrain = len(trainloader)
	losses, train_accuracy, test_accuracy = [], [], []
	lambdaloss = lambdaloss0
	i = 0
	print('---train')
	for epoch in range(1, nepochs + 1):
		model.train()
		loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
		for x, y in trainloader:
			i += 1
			optimizer.zero_grad()
			out, rs = model(x)
			if us == 0:
				loss = criterion(out, y) + lambdatransport * sum([torch.norm(r, 2) for r in rs])
			else:
				loss = lambdaloss * criterion(out, y) + lambdatransport * sum([torch.norm(r, 2) for r in rs])
			_, pred = torch.max(out.data, 1)
			update_meters(y, pred, loss.item(), loss_meter, accuracy_meter)
			loss.backward()
			optimizer.step()
			if us > 0 and i % us == 0:
				out, rs = model(x)
				lambdaloss += tau * criterion(out, y).item()
		epochloss = loss_meter.avg
		epochacc = accuracy_meter.avg
		acc, F, C, W = test(model, criterion, lambdatransport, lambdaloss, testloader, X_test, Y_test, epoch, nblocks, folder)
		print('[epoch %d] lambda loss: %.3f train loss: %.3f train accuracy: %.3f test accuracy: %.3f' % (epoch, lambdaloss, epochloss, epochacc, acc))
		losses.append(epochloss)
		train_accuracy.append(epochacc)
		test_accuracy.append(acc)
		if epoch > 3 and test_accuracy[-1] == test_accuracy[-2] == test_accuracy[-3] == 1 and train_accuracy[-1] == train_accuracy[-2] == train_accuracy[-3] == 1:
			break
	return losses, test_accuracy, epoch, F, C, W

def test(model, criterion, lambdatransport, lambdaloss, testloader, X_test, Y_test, epoch, nblocks, folder):
	model.eval()
	X = []
	loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
	global inps, outs, gouts
	inps, outs, gouts = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
	for (x, y) in testloader:
		out, rs = model(x)
		loss = lambdaloss * criterion(out, y) + lambdatransport * sum([torch.norm(r, 2) for r in rs])
		loss.backward()
		_, pred = torch.max(out.data, 1)
		update_meters(y, pred, loss.item(), loss_meter, accuracy_meter)
	inps_ = {name: np.vstack(inp) for name, inp in inps.items()}
	outs_ = {name: np.vstack(out) for name, out in outs.items()}
	gouts_ = {name: np.vstack(gout) for name, gout in gouts.items()}
	F, C, W = plotmetrics(outs_, X_test, Y_test, gouts_, epoch, model, nblocks, folder)
	return accuracy_meter.avg, F, C, W

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

def plotmetrics(outs_, X_test, Y_test, gouts_, epoch, model, nblocks, folder):
	F, C, W = [], [], []
	a, b, c, d = model.fcOut.weight[0, 0].item(), model.fcOut.weight[0, 1].item(), model.fcOut.weight[1, 0].item(), model.fcOut.weight[1, 1].item()
	e, f = model.fcOut.bias[0].item(), model.fcOut.bias[1].item()
	g, h = (c - a) / (b - d), (f - e) / (b - d)
	if nblocks < 10 : # point cloud
		x = np.linspace(- 20, 10, 100)
		y = g * x + h
		fig, ax = plt.subplots(3, 3, sharex = 'all', sharey = 'all')
		fig.set_size_inches(18, 18)
		fig.suptitle('Transformed test data after epoch {}. Linear classifier slope {} intercept {}'.format(epoch, g, h))
	for i in range(nblocks):
		X = outs_['blocks.' + str(i)]
		if nblocks < 10 : # point cloud
			row, col = int(i / 3), int(i % 3)
			ax[row, col].scatter(X[Y_test == 0, 0], X[Y_test == 0, 1], c = 'red')
			ax[row, col].scatter(X[Y_test == 1, 0], X[Y_test == 1, 1], c = 'blue')
			ax[row, col].plot(x, y, '-g', label = 'linear classifier')
			ax[row, col].set_title('block ' + str(i + 1))
		X_ = X_test if i == 0 else outs_['blocks.' + str(i - 1)] 
		W.append(W2(X_, X)) # W2 movement
		X = outs_['blocks.' + str(i) + '.fc2']
		F.append(np.mean(np.sqrt(np.sum(np.abs(X) ** 2, axis = -1)))) # forcing function
		if i > 0 : # cosine loss
			L = gouts_['blocks.' + str(i - 1)]
			C.append(np.mean(np.sum(np.multiply(X, L), axis = -1)))
	if nblocks < 10 : # plot point cloud
		fig.savefig(os.path.join(folder, 'testset_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
		plt.close(fig)
	plot_arrays(F, C, W, nblocks, epoch, folder)
	return F, C, W

def plot_arrays(F, C, W, nblocks, epoch, folder):
	plt.figure(figsize = (7, 7)) # plot cosine loss
	plt.plot(list(range(2, nblocks + 1)), C)
	plt.title('cos(f, grad L) after epoch ' + str(epoch))
	plt.xlabel('block $k$')
	plt.ylabel('cos( f(h), grad_h L )')
	plt.savefig(os.path.join(folder, 'cos_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()
	plt.figure(figsize = (7, 7)) # plot forcing function and W2 movement
	plt.plot(list(range(1, nblocks + 1)), F, 'b', label = 'Average $|| f_k(x) ||$')
	plt.plot(list(range(1, nblocks + 1)), W, 'r', label = '$W_2$ distance')
	plt.title('f and wasserstein distance after epoch ' + str(epoch))
	plt.xlabel('block $k$')
	plt.legend(loc = 'best')
	plt.savefig(os.path.join(folder, 'distance_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()

def experiment(ndata = 1000, 
			   testsize = 0.2, 
			   data = 1,
			   noise = 0.05, 
			   factor = 0.3, 
			   dataseed = None,
			   modelseed = None,
			   nblocks = 9,
			   inputdim = 2,
			   hiddendim = 2, 
			   batchnorm = False,
			   nclasses = 2,
			   learnclassifier = False,
			   yintercept = 20,
			   biginit = False,
			   biginitstd = 5,
			   lambdatransport = 1,
			   lambdaloss0 = 0.1,
			   tau = 0.1,
			   us = 5,
			   batchsize = 10,
			   nepochs = 100,
			   learningrate = 0.01, 
			   beta1 = 0.9,
			   beta2 = 0.99,
			   experiments = False) :
	t0 = time.time()
	folder = makefolder(ndata, testsize, data, noise, factor, dataseed, modelseed, nblocks, inputdim, hiddendim, batchnorm, nclasses,
						learnclassifier, yintercept, biginit, biginitstd, lambdatransport, lambdaloss0, tau, us, batchsize, nepochs, learningrate, beta1, beta2)
	if experiments : 
		stdout0 = sys.stdout
		sys.stdout = open(os.path.join(folder, 'output.txt'), 'wt')
	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('--- experiment from cloud3.py with parameters')
	for name in names:
		print('%s = %s' % (name, values[name]))
	if us == 0 and (lambdaloss0 != 1 or tau > 0):
		print('us = 0 means no uzawa. lambda loss is fixed to 1 and tau to 0')
		lambdaloss0, tau = 1, 0
	if us > 0 and lambdatransport != 1:
		print('us > 0 means uzawa. lambda transport is fixed to 1')
		lambdatransport = 1
	trainloader, testloader, X, Y, X_test, Y_test = dataloaders(ndata, testsize, data, noise, factor, dataseed, batchsize)
	plotdata(X, Y, learnclassifier, yintercept, 'data', folder)
	plotdata(X_test, Y_test, learnclassifier, yintercept, 'testdata', folder)
	if modelseed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(modelseed)
		np.random.seed(modelseed)
	initialize = partial(initialize_, biginit, biginitstd)
	model = OneRepResNet(nblocks, inputdim, hiddendim, batchnorm, nclasses, learnclassifier, yintercept, initialize)
	print('--- model', model)
	global inps, outs, gouts
	inps, outs, gouts = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
	register_hooks(model)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = learningrate, betas = (beta1, beta2))
	losses, accuracy, epochs, F, C, W = train(model, nepochs, criterion, lambdatransport, lambdaloss0, tau, us, optimizer, trainloader, testloader, X_test, Y_test, nblocks, folder)
	plotscores(losses, accuracy, folder)
	t1 = time.time() - t0
	print('--- experiment time %.1f s' % (t1))
	plot_border(X, Y, model, folder, batchnorm)
	if experiments:
		print('--- losses \n', losses, '\n--- accuracy \n', accuracy, '\n--- C \n', C, '\n--- W \n', W, '\n--- F \n', F)
		sys.stdout.close()
		sys.stdout = stdout0
		return accuracy, epochs, F, C, W, t1

def plot_border(X, Y, model, folder, batchnorm):
	model.eval()
	cmap = colors.ListedColormap(['orange', 'cyan'])
	bounds = [0, 0.5, 1]
	norm = colors.BoundaryNorm(bounds, cmap.N)
	n = 200
	M = np.zeros((n, n))
	xx, yy = np.linspace(-2, 2, n), np.linspace(-2, 2, n)
	xv, yv = np.meshgrid(xx, yy)
	for i in range(n):
		for j in range(n):
			x = torch.tensor([[xv[i, j], yv[i, j]]])
			y, _ = model(x)
			_, pred = torch.max(y.data, 1)
			M[i, j] = pred.item()
	fig, ax = plt.subplots()
	ax.imshow(M, cmap = cmap, norm = norm)
	X = n * X / 4 + n / 2
	plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c = 'red')
	plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c = 'blue')
	plt.savefig(os.path.join(folder, 'border.png'), bbox_inches = 'tight')
	plt.close()


def experiments(parameters, average):
	t0, j, f = time.time(), 0, 110
	sep = '-' * f 
	if average:
		accuracies, epochs, F, C, W = [], [], [], [], [] 
	nparameters = len(parameters)
	nexperiments = int(np.prod([len(parameters[i][1]) for i in range(nparameters)]))
	print('\n' + sep, 'cloud3.py')
	print(sep, nexperiments, 'experiments ' + ('to average ' if average else '') + 'over parameters:')
	pprint.pprint(parameters, width = f, compact = True)
	for params in product([values for name, values in parameters]) :
		j += 1
		print('\n' + sep, 'experiment %d/%d with parameters:' % (j, nexperiments))
		pprint.pprint([parameters[i][0] + ' = ' + str(params[i]) for i in range(nparameters)], width = f, compact = True)
		accuracy_, epochs_, F_, C_, W_, t1 = experiment(*params, True)
		if average:
			accuracies.append(np.max(accuracy_))
			epochs.append(epochs_)
			F.append(F_)
			C.append(C_)
			W.append(W_)
		print(sep, 'experiment %d/%d over. took %.1f s. total %.1f s' % (j, nexperiments, t1, time.time() - t0))
	if average:
		avg = lambda x : np.mean(np.vstack(x), axis = 0)
		acc, epochs, F, C, W = np.mean(accuracies), np.mean(epochs), avg(F), avg(C), avg(W)
		confint = st.t.interval(0.95, len(accuracies) - 1, loc = acc, scale = st.sem(accuracies))
		print('\naverage epochs', epochs)
		print('\nall best acc', accuracies)
		print('\naverage best acc', acc)
		print('\nconfint', confint)
		folder = 'avg_exp' + ''.join(map(str, [p[1][0] for p in parameters]))
		os.makedirs(folder)
		plot_arrays(F, C, W, parameters[7][1][0], epochs, folder)
	print(('\n' if not average else '') + sep, 'total time for %d experiments: %.1f s' % (j, time.time() - t0))

def product(iterables):
    if len(iterables) == 0 :
        yield ()
    else :
        it = iterables[0]
        for item in it :
            for items in product(iterables[1: ]) :
                yield (item, ) + items


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-nd", "--ndata", type = int, default = [1000], nargs = '*')
	parser.add_argument("-ts", "--testsize", type = float, default = [0.2], nargs = '*')
	parser.add_argument("-da", "--data", type = int, default = [1], choices = [1, 2], nargs = '*')
	parser.add_argument("-no", "--noise", type = float, default = [0.05], nargs = '*')
	parser.add_argument("-fa", "--factor", type = float, default = [0.3], nargs = '*')
	parser.add_argument("-ds", "--dataseed", type = int, default = [None], nargs = '*')
	parser.add_argument("-ms", "--modelseed", type = int, default = [None], nargs = '*')
	parser.add_argument("-nb", "--nblocks", type = int, default = [9], nargs = '*')
	parser.add_argument("-dd", "--datadim", type = int, default = [2], nargs = '*')
	parser.add_argument("-hd", "--hiddendim", type = int, default = [2], nargs = '*')
	parser.add_argument("-bn", "--batchnorm", type = int, default = [0], nargs = '*')
	parser.add_argument("-nc", "--nclasses", type = int, default = [2], nargs = '*')
	parser.add_argument("-lc", "--learnclassifier", type = int, default = [0], nargs = '*')
	parser.add_argument("-yi", "--yintercept", type = int, default = [20], nargs = '*')
	parser.add_argument("-bi", "--biginit", type = int, default = [0], nargs = '*')
	parser.add_argument("-sd", "--biginitstd", type = float, default = [1.0], nargs = '*')
	parser.add_argument("-lt", "--lambdatransport", type = float, default = [1.], nargs = '*')
	parser.add_argument("-ll", "--lambdaloss0", type = float, default = [0.1], nargs = '*')
	parser.add_argument("-ta", "--tau", type = float, default = [0.1], nargs = '*')
	parser.add_argument("-us", "--uzawasteps", type = int, default = [5], nargs = '*')
	parser.add_argument("-bs", "--batchsize", type = int, default = [10], nargs = '*')
	parser.add_argument("-ne", "--nepochs", type = int, default = [100], nargs = '*')
	parser.add_argument("-lr", "--learningrate", type = float, default = [0.01], nargs = '*')
	parser.add_argument("-b1", "--beta1", type = float, default = [0.9], nargs = '*')
	parser.add_argument("-b2", "--beta2", type = float, default = [0.99], nargs = '*')
	parser.add_argument("-ex", "--experiments", action = 'store_true')
	parser.add_argument("-av", "--averageexperiments", action = 'store_true')
	args = parser.parse_args()
	
	if args.experiments or args.averageexperiments:
		parameters = [(name, values) for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiments(parameters, args.averageexperiments)
	else :
		parameters = [values[0] for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiment(*parameters, False)
	
	






	









