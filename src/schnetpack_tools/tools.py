import uuid
import torch
import shutil
import numpy as np

from torch.optim import Adam

from schnetpack import train_test_split
from schnetpack.representation import SchNet
from schnetpack.data import AtomsData, AtomsLoader
from schnetpack.train import Trainer, EarlyStoppingHook
from schnetpack.train import MeanAbsoluteError, CSVHook
from schnetpack.atomistic import Atomwise, AtomisticModel

from schnetpack_tools.metrics import R2Score
from schnetpack_tools.environment import OpenCLEnvironmentProvider

from schnetpack.utils import load_model

class IterativeDatasetReduction():
	"""docstring for IterativeDatasetReduction"""
	def __init__(
		self,
		dbpath,
		lr,
		batch_size,
		num_workers,
		loss_fn,
		n_atom_basis=128,
		n_layers=2,
		n_filters=128,
		n_interactions=3,
		cutoff=5.0,
		n_gaussians=25,
		frac=0.02,
		shm=True
	):

		### SchNet settings ###
		self.lr = lr

		self.batch_size = batch_size
		self.num_workers = num_workers

		self.n_atom_basis = n_atom_basis
		self.n_layers = n_layers

		self.n_filters = n_filters
		self.n_interactions = n_interactions

		self.cutoff = cutoff
		self.n_gaussians = n_gaussians

		self.properties = ['energy', 'forces', 'stress']
		self.loss_fn = loss_fn

		#######################

		self.i = 0
		self.frac = frac

		if shm:
			dbcopy = '/dev/shm/' + uuid.uuid4().hex + '.db'
			shutil.copyfile(dbpath, dbcopy)

			self.dataset = AtomsData(dbcopy,
				load_only=self.properties,
				environment_provider=OpenCLEnvironmentProvider(self.cutoff, 0),
				centering_function=None)
		else:
			self.dataset = AtomsData(dbpath,
				load_only=self.properties,
				environment_provider=OpenCLEnvironmentProvider(self.cutoff, 0),
				centering_function=None)

		self.idx_rem = np.arange(len(self.dataset))
		np.random.shuffle(self.idx_rem)

		I = np.arange(round(self.frac * len(self.idx_rem)))

		self.idx_red = self.idx_rem[I]
		self.idx_rem = np.delete(self.idx_rem, I)

	def evaluate_fn(self, batch, result, fid=None):
		with torch.no_grad():

			N = torch.sum(batch['_atom_mask'], 1)

			E_err = torch.abs(batch['energy'] - result['energy']).view(-1) / N
			F_err = torch.sum(torch.abs(batch['forces'] - result['forces']), (2,1)) / N
			S_err = torch.mean(torch.abs(batch['stress'] - result['stress']), (2,1))

			if fid is not None:
				for e, f, s in zip(E_err, F_err, S_err):
					fid.write('%f,%f,%f\n' % (e, f, s))

		return ((E_err > 0.1).byte() + (F_err > 1.0).byte() + (S_err > 0.05).byte() > 0)

	def train(self, device, n_epochs, patience=100, threshold_ratio=0.0001):

		self.i += 1

		reduced = self.dataset.create_subset(self.idx_red)

		num_val = round(0.10 * len(reduced))
		train, val, test = train_test_split(
			data=reduced,
			num_train=len(reduced) - num_val,
			num_val=num_val)

		train_loader = AtomsLoader(train,
			batch_size=round(self.batch_size),
			num_workers=self.num_workers,
			shuffle=True, pin_memory=True)

		val_loader = AtomsLoader(val,
			batch_size=round(self.batch_size/2),
			num_workers=self.num_workers,
			pin_memory=True)

		representation = SchNet(
			n_atom_basis=self.n_atom_basis,
			n_filters=self.n_filters,
			n_interactions=self.n_interactions,
			cutoff=self.cutoff,
			n_gaussians=self.n_gaussians)

		output_modules = Atomwise(representation.n_atom_basis,
			n_layers=self.n_layers,
			property='energy',
			derivative='forces',
			stress='stress',
			negative_dr=True,
			create_graph=True)

		model = AtomisticModel(representation, output_modules)

		optimizer = Adam(model.parameters(), lr=self.lr)

		hooks = [
		CSVHook('log_%i' % self.i, [
			MeanAbsoluteError('energy', 'energy'),
			MeanAbsoluteError('forces', 'forces', element_wise=True),
			MeanAbsoluteError('stress', 'stress'),
			R2Score('energy', 'energy'),
			R2Score('forces', 'forces', element_wise=True),
			R2Score('stress', 'stress')],
			every_n_epochs=1)
		]

		hooks.append(EarlyStoppingHook(patience, threshold_ratio))

		trainer = Trainer('output_%i/' % self.i, model, self.loss_fn, optimizer, train_loader, val_loader,
			hooks=hooks, keep_n_checkpoints=1, checkpoint_interval=1000000)

		print('Running training!')
		print('    Reduced images: %i' % len(reduced))
		print('    Traning images: %i' % len(train))
		print(' Validation images: %i' % len(val))
		print('')

		trainer.train(device, n_epochs)

	def evaluate(self, device, log_remaining=True):

		model = load_model('output_%i/best_model' % self.i, map_location=device)
		model.output_modules[0].create_graph = False

		remaining = self.dataset.create_subset(self.idx_rem)

		loader = AtomsLoader(remaining,
			batch_size=round(self.batch_size),
			num_workers=self.num_workers,
			pin_memory=True)

		print('Running evaluation!')

		if log_remaining:
			fid = open('log_%i/remaining.csv' % self.i, 'w')
			fid.write('Energy (eV),Force (eV/Å),Stress (eV/Å³)\n')
		else:
			fid = None

		passfail = []
		for batch in loader:
			batch = {k: v.to(device) for k, v in batch.items()}
			result = model(batch)
			passfail += self.evaluate_fn(batch, result, fid).tolist()

		fid.close()

		I = np.where(passfail)[0]
		percentage = 100 * len(I) / len(self.idx_rem)

		if percentage > 5.0:
			np.random.shuffle(I)
			J = I[0:round(self.frac * len(I))]

			self.idx_red = np.append(self.idx_red, self.idx_rem[J])
			self.idx_rem = np.delete(self.idx_rem, J)
		else:
			1+1

		print('           Failed images: %i' % len(I))
		print('            Added images: %i' % len(J))
		print(' Percentage of remaining: %6.2f%' % percentage)
		print('Reduced/Remaining images: %i/%i' % (len(self.idx_red), len(self.idx_rem)))
		print('')

	def reduce(self, device, n_epochs, patience=100, threshold_ratio=0.0001, log_remaining=True):

		while True:
			self.train(device, n_epochs, patience, threshold_ratio)
			self.evaluate(device, log_remaining)
