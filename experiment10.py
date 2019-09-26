# validation set
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
	print('-' * 64, 'decoder', encoder)
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
	def __init__(self, filters, batchnorm, bias):
		super(FirstResBlock, self).__init__()
		self.batchnorm = batchnorm
		self.cv1 = nn.Conv2d(filters, filters, 3, 1, 1, bias = bias)
		if self.batchnorm:
			self.bn2 = nn.BatchNorm2d(filters)
		self.cv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = self.cv1(x)
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + z, z 

class ResBlock(nn.Module):
	def __init__(self, filters, batchnorm, bias):
		super(ResBlock, self).__init__()
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
		return x + z, z

class OneRepResNet(nn.Module):
	def __init__(self, datashape, nclasses, filters, nblocks, batchnorm, bias, encoder, initname = 'orthogonal', initgain = 0.01,
				 learn_cl = True, cl_name = '3Lin', cl_file = None):
		super(OneRepResNet, self).__init__()
		self.classifier_name = cl_name
		self.encoder = encoder
		for param in self.encoder.parameters():
			param.requires_grad = False
		self.stage1 = nn.ModuleList([FirstResBlock(filters, batchnorm, bias) if i == 0 else ResBlock(filters, batchnorm, bias) for i in range(nblocks)])
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
		rs, xs = [], [x]
		for block in self.stage1:
			x, r = block(x)
			rs.append(r)
			xs.append(x)
		if self.classifier_name[-3:] == 'Lin':
			x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs, xs

def cl_input_hook(name, f, fn, rat_meters, for_meters, cos_meters, clinps, mod, inp, out):
	inp0 = inp[0] if type(inp) is tuple else inp
	clinps.append(inp0.cpu().detach().numpy())

def block_input_hook(name, f, fn, rat_meters, for_meters, cos_meters, clinps, mod, inp, out):
	i_block = int(name[7:]) 
	h_ = inp[0] if type(inp) is tuple else inp
	hn_ = l2norm(h_.cpu().detach().numpy().copy())
	fn_ = fn.get(name + '.cv2')
	ratio = fn_ / np.where(hn_ == 0, 0.001, hn_)
	rat_meters[i_block].update(np.mean(ratio), ratio.shape[0])
	
def forcing_output_hook(name, f, fn, rat_meters, for_meters, cos_meters, clinps, mod, inp, out):
	f_ = out[0] if type(out) is tuple else out
	f_ = f_.cpu().detach().numpy().copy()
	fn_ = l2norm(f_)
	f[name], fn[name] = f_, fn_
	i_block = int(name[7]) if name[8] == '.' else int(name[7:9])
	for_meters[i_block].update(np.mean(fn_), fn_.shape[0])

def block_output_hook(name, f, fn, rat_meters, for_meters, cos_meters, clinps, mod, inp, out):
	o = out[0] if type(out) is tuple else out
	o = o.cpu().detach().numpy().copy()

def outgrad_hook(name, f, fn, rat_meters, for_meters, cos_meters, clinps, mod, ginp, gout):
	g_ = gout[0] if type(gout) is tuple and gout[0] is not None else gout
	g_ = g_.cpu().detach().numpy().copy()
	gn_ = l2norm(g_)
	if name == 'encoder':
		i_block = 0
	else :
		i_block = int(name[7:]) + 1
	s = 'stage1.' + str(i_block) + '.cv2'
	f_, fn_ = f[s], fn[s]
	cos = np.sum(np.multiply(f_, g_), axis = (1, 2, 3))
	cos = cos / np.multiply(np.where(fn_ == 0, 0.001, fn_), np.where(gn_ == 0, 0.001, gn_))
	cos_meters[i_block].update(np.mean(cos), cos.shape[0])

def register_hooks(model, block_input_hook, block_output_hook, forcing_output_hook, outgrad_hook, cl_input_hook, *args):
	for name, m in model.named_modules():
		if name == 'encoder' :
			m.register_backward_hook(partial(outgrad_hook, name, *args))
		elif len(name) in [8, 9, 10] and name[0:5] == 'stage' :
			m.register_forward_hook(partial(block_input_hook, name, *args))
			m.register_forward_hook(partial(block_output_hook, name, *args))
			if int(name[7:]) < nblocks - 1:
				m.register_backward_hook(partial(outgrad_hook, name, *args))
		if len(name) in [12, 13, 14] and name[0:5] == 'stage' and name[-3:] == 'cv2' :
			m.register_forward_hook(partial(forcing_output_hook, name, *args))
		if name == 'classifier' :
			m.register_forward_hook(partial(cl_input_hook, name, *args))

def train(model, optimizer, scheduler, criterion, trainloader, valloader, testloader, decoder, mean, std, lambdatransport, lambdaloss0, tau, uzawa_steps, folder, 
		  clip = 0, nepochs = 30):
	target_acc = 0.994 if dataset_name == 'mnist' else (0.93 if dataset_name == 'cifar10' else 0.74)
	train_loss, train_acc, val_loss, val_acc = [], [], [], []
	lambdaloss = lambdaloss0
	t0, it = time.time(), 0
	print('--- Begin trainning')
	for e in range(nepochs):
		t1 = time.time()
		model.train()
		loss_meter, acc_meter, time_meter = AverageMeter(), AverageMeter(), AverageMeter()
		for j, (x, y) in enumerate(trainloader):
			t2, it = time.time(), it + 1
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			out, rs, xs = model(x)
			loss = lambdaloss * criterion(out, y) + lambdatransport * sum([torch.mean(r ** 2) for r in rs])
			loss.backward()
			if clip > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()
			_, pred = torch.max(out.data, 1)
			update_meters(y = y, pred = pred, loss = loss.item(), loss_meter = loss_meter, acc_meter = acc_meter, t = time.time() - t2, time_meter = time_meter)
			if uzawa_steps > 0 and it % uzawa_steps == 0:
				out, rs, xs = model(x)
				lambdaloss += tau * criterion(out, y).item()
			if j % 100 == 0 :
				metrics = (e + 1, nepochs, j + 1, len(trainloader), lambdaloss, loss_meter.avg, acc_meter.avg, time_meter.avg, lambdaloss)
				print('Epoch {}/{} Batch {}/{} | Lambda loss {:.4f} Train loss {:.4f} Train acc {:.4f} Avg batch time {:.4f}s'.format(*metrics))
		scheduler.step()
		train_loss.append(loss_meter.avg)
		train_acc.append(acc_meter.avg)
		val_loss_, val_acc_ = validate(model, criterion, optimizer, valloader, lambdatransport, lambdaloss)
		val_loss.append(val_loss_)
		val_acc.append(val_acc_)
		t3 = time.time()
		print('Epoch {}/{} over | Val acc {:.4f} Epoch time {:.4f}s Total time {:.4f}s'.format(e + 1, nepochs, val_acc_, t3 - t1, t3 - t0))
		if e > 3 and val_acc[-1] > target_acc and val_acc[-2] > target_acc and val_acc[-3] > target_acc:
			break
	test_loss, test_acc, test_transport = test(model, criterion, optimizer, testloader, decoder, mean, std, lambdatransport, lambdaloss, e, folder)
	print('Final test acc {:.4f} | Transport {:.4f}'.format(test_acc, test_transport))
	return train_loss, val_acc, test_acc, test_transport

def validate(model, criterion, optimizer, valloader, lambdatransport, lambdaloss):
	model.eval()
	loss_meter, acc_meter = AverageMeter(), AverageMeter()
	for j, (x, y) in enumerate(valloader):
		with torch.no_grad():
			optimizer.zero_grad()
			x, y = x.to(device), y.to(device)
			out, rs, xs = model(x)
			loss = lambdaloss * criterion(out, y) + lambdatransport * sum([torch.mean(r ** 2) for r in rs])
			_, pred = torch.max(out.data, 1)
			update_meters(y, pred, loss.item(), loss_meter, acc_meter)
	return loss_meter.avg, acc_meter.avg

def test(model, criterion, optimizer, testloader, decoder, mean, std, lambdatransport, lambdaloss, e, folder, dist = True, tsne = False):
	model.eval()
	loss_meter, acc_meter, trs_meter = AverageMeter(), AverageMeter(), AverageMeter()
	f, fn, clinps = dict(), dict(), []
	rat_meters = {i : AverageMeter() for i in range(nblocks)}
	for_meters = {i : AverageMeter() for i in range(nblocks)}
	cos_meters = {i : AverageMeter() for i in range(nblocks)}
	register_hooks(model, block_input_hook, block_output_hook, forcing_output_hook, outgrad_hook, cl_input_hook, f, fn, rat_meters, for_meters, cos_meters, clinps)
	idx_batch = np.random.choice(len(testloader))
	for j, (x, y) in enumerate(testloader):
		optimizer.zero_grad()
		x, y = x.to(device), y.to(device)
		out, rs, xs = model(x)
		transport = sum([torch.mean(r ** 2) for r in rs])
		loss = lambdaloss * criterion(out, y) + lambdatransport * transport
		loss.backward()
		_, pred = torch.max(out.data, 1)
		update_meters(y = y, pred = pred, loss = loss.item(), loss_meter = loss_meter, acc_meter = acc_meter, transport = transport.item(), transport_meter = trs_meter)
		if j == idx_batch:
			idx_images = np.random.choice(x.size()[0], 5, replace = False)
			zs = [decoder(xi.detach()) for xi in xs]
			images = [x.cpu().detach().numpy().copy()[idx_images, :, :, :]] + [zi.cpu().detach().numpy().copy()[idx_images, :, :, :] for zi in zs]
			show_decoded_images(images, mean, std, os.path.join(folder, 'decodings-epoch{}.png'.format(e)))
			xs_np = [xi.cpu().detach().numpy().copy() for xi in xs]
			distances = [emd(xs_np[i - 1], xs_np[i]) for i in range(1, len(xs_np))] if dist else None
			y_np = y.cpu().detach().numpy().copy()
			if tsne:
				for k, xi in enumerate(xs_np):
					tsne = TSNE(n_components = 2, perplexity = 30.0, init = 'pca', verbose = 0)
					emb = tsne.fit_transform(xi.reshape((xi.shape[0], -1)))
					convex_hulls= convexHulls(emb, y_np) 
					ellipses = best_ellipses(emb, y_np)
					nh = neighboring_hit(emb, y_np)
					Visualization(emb, y_np, convex_hulls, ellipses, nh, os.path.join(folder, 'tsne-epoch{}-block{}.png'.format(e, k))) 
	ratios, forcings, cosines = get_avg(rat_meters, nblocks), get_avg(for_meters, nblocks), get_avg(cos_meters, nblocks)
	np.save(os.path.join(folder, 'clinps.npy'), clinps)
	plot_arrays(ratios, cosines, forcings, distances, nblocks, e, folder)
	return loss_meter.avg, acc_meter.avg, trs_meter.avg

def experiment(dataset_name, filters, learningrate, lambdatransport, lambdaloss0, tau, uzawasteps, batchnorm, bias, clip, learnclassifier, nblocks, 
			   nepochs, initname, initgain, classifier_name, trainsize, split = 0.5, seed = None, experiments = False):

	t0 = time.time()
	if seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		np.random.seed(seed)
	
	trainloader, valloader, testloader, datashape, nclasses, mean, std = dataloaders(dataset_name, batchsize, trainsize, split)
	folder = dataset_name + 'onerep' + str(filters) + 'exp' + time.strftime("%Y%m%d-%H%M%S") + '-lt' + str(lambdatransport) + '-ll' + str(lambdaloss0)
	make_folder(folder)
	if nepochs > 20:
		stdout0 = sys.stdout
		sys.stdout = open(os.path.join(folder, 'log.txt'), 'wt')
	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('--- experiment from experiment9.py with parameters')
	for name in names:
		print('%s = %s' % (name, values[name]))
	if uzawasteps == 0 and (lambdaloss0 != 1 or tau > 0):
		print('us = 0 means no transport loss. lambda loss is fixed to 1, tau to 0, and lambda transport to 0')
		lambdaloss0, tau, lambdatransport = 1, 0, 0
	if uzawasteps > 0 and lambdatransport != 1:
		print('us > 0 means uzawa. lambda transport is fixed to 1')
		lambdatransport = 1

	ae_penalty, ae_lambda, ae_folder = 0, 0, 3
	(ae_ds, ae_bn) = (False, True) if dataset_name == 'cifar100' else (True, False)
	encoder_file, decoder_file = pretrained_ae_files(filters, ae_ds, ae_bn, ae_penalty, ae_lambda, ae_folder)
	encoder, decoder = load_autoencoder(datashape[1], filters, ae_ds, ae_bn, encoder_file, decoder_file)
	test_autoencoder(datashape, filters, ae_ds, encoder, decoder, testloader, mean, std)
	encoder.to(device)
	decoder.to(device)
	
	model = OneRepResNet(datashape, nclasses, filters, nblocks, batchnorm, bias, encoder, initname, initgain, learnclassifier, classifier_name)
	model.to(device)
	print(model)
	summary(model, datashape[1:])
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = learningrate, momentum = 0.9, weight_decay = 0.0001)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [120, 160, 200], gamma = 0.2)
	
	train_loss, val_acc, test_acc, test_transport = train(model, optimizer, scheduler, criterion, trainloader, valloader, testloader, decoder, mean, std, 
					  					  				  lambdatransport, lambdaloss0, tau, uzawasteps, folder, clip, nepochs)

	if experiments and nepochs > 5:
		print('--- train losse \n', train_loss, '\n--- val accuracy \n', val_acc, '\n--- test accuracy \n', test_acc)
		sys.stdout.close()
		sys.stdout = stdout0

	if not experiments:
		torch.save(model.classifier.state_dict(), os.path.join(folder, 'classifierweights.pth'))
	del model
	return train_loss, val_acc, test_acc, test_transport, time.time() - t0

def experiments(parameters, average, transport):
	t0, j, f = time.time(), 0, 110
	sep = '-' * f 
	accuracies, transports = [], []
	nparameters = len(parameters)
	nexperiments = int(np.prod([len(parameters[i][1]) for i in range(nparameters)]))
	print('\n' + sep, 'experiment9.py')
	print(sep, nexperiments, 'experiments ' + ('to average ' if average else '') + 'over parameters:')
	pprint.pprint(parameters, width = f, compact = True)
	for params in product([values for name, values in parameters]) :
		j += 1
		print('\n' + sep, 'experiment %d/%d with parameters:' % (j, nexperiments))
		pprint.pprint([parameters[i][0] + ' = ' + str(params[i]) for i in range(nparameters)], width = f, compact = True)
		train_loss, val_acc, test_acc, test_trs, t1 = experiment(*params, True)
		accuracies.append(test_acc)
		transports.append(test_trs)
		print(sep, 'experiment %d/%d over. took %.1f s. total %.1f s' % (j, nexperiments, t1, time.time() - t0))
	acc = np.mean(accuracies)
	confint = st.t.interval(0.95, len(accuracies) - 1, loc = acc, scale = st.sem(accuracies))
	print('\nall test acc', accuracies)
	print('\naverage test acc', acc)
	print('\nconfint', confint)
	if transport:
		trs = np.mean(transports)
		confint = st.t.interval(0.95, len(transports) - 1, loc = trs, scale = st.sem(transports))
		print('\nall test trs', transports)
		print('\naverage test trs', trs)
		print('\nconfint', confint)
	print(('\n' if not average else '') + sep, 'total time for %d experiments: %.1f s' % (j, time.time() - t0))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-dat", "--dataset", default = ['mnist'], choices = ['mnist', 'cifar10', 'cifar100'], nargs = '*')
	parser.add_argument("-fil", "--filters", type = int, default = [32], choices = [32, 64, 100, 150], nargs = '*')
	parser.add_argument("-lrr", "--learningrate", type = float, default = [0.01], nargs = '*')
	parser.add_argument("-lmt", "--lambdatransport", type = float, default = [1], nargs = '*')
	parser.add_argument("-lml", "--lambdaloss0", type = float, default = [5] , nargs = '*')
	parser.add_argument("-tau", "--tau", type = float, default = [1], nargs = '*')
	parser.add_argument("-uzs", "--uzawasteps", type = int, default = [5], nargs = '*')
	parser.add_argument("-btn", "--batchnorm", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-bia", "--bias", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-clp", "--clip", type = float, default = [0], nargs = '*')
	parser.add_argument("-lcl", "--learnclassifier", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-nbl", "--nblocks", type = int, default = [9], nargs = '*')
	parser.add_argument("-nep", "--nepochs", type = int, default = [30], nargs = '*')
	parser.add_argument("-inn", "--initname", default = ['orthogonal'], choices = ['orthogonal', 'normal', 'kaiming'], nargs = '*')
	parser.add_argument("-ing", "--initgain", type = float, default = [0.01], nargs = '*')
	parser.add_argument("-cla", "--classifier", default = ['3Lin'], choices = ['1Lin', '2Lin', '3Lin'], nargs = '*')
	parser.add_argument("-trs", "--trainsize", type = int, default = [0], nargs = '*')
	parser.add_argument("-spl", "--split", type = float, default = [0.5], nargs = '*')
	parser.add_argument("-see", "--seed", type = int, default = [None], nargs = '*')
	parser.add_argument("-exp", "--experiments", action = 'store_true')
	parser.add_argument("-avg", "--averageexperiments", action = 'store_true')
	parser.add_argument("-tra", "--transportexperiments", action = 'store_true')
	args = parser.parse_args()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	batchsize = 128
	filters = args.filters[0]
	dataset_name = args.dataset[0]
	nblocks = args.nblocks[0]

	if args.experiments or args.averageexperiments or args.transportexperiments:
		parameters = [(name, values) for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments', 'transportexperiments']]
		experiments(parameters, args.averageexperiments, args.transportexperiments)
	else :
		parameters = [values[0] for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments', 'transportexperiments']]
		experiment(*parameters, False)
	

	



