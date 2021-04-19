import torch
import numpy as np

from schnetpack import Properties
from schnetpack.train import Metric

class MeanSquaredError(Metric):
	r"""
	Metric for mean square error. For non-scalar quantities, the mean of all
	components is taken.
	Args:
		target (str): name of target property
		model_output (int, str): index or key, in case of multiple outputs
			(Default: None)
		name (str): name used in logging for this metric. If set to `None`,
			`MSE_[target]` will be used (Default: None)
		element_wise (bool): set to True if the model output is an element-wise
			property (forces, positions, ...)
		atom_wise (bool): set to True to divide the model output by the number
			of atoms to get eV per atom, etc.
	"""

	def __init__(
		self,
		target,
		model_output=None,
		bias_correction=None,
		name=None,
		element_wise=False,
		atom_wise=False
	):
		name = "MSE_" + target if name is None else name
		super(MeanSquaredError, self).__init__(
			target=target,
			model_output=model_output,
			name=name,
			element_wise=element_wise,
		)

		self.bias_correction = bias_correction
		self.atom_wise = atom_wise

		self.l2loss = 0.0
		self.n_entries = 0.0

	def reset(self):
		"""Reset metric attributes after aggregation to collect new batches."""
		self.l2loss = 0.0
		self.n_entries = 0.0

	def _get_diff(self, y, yp):
		diff = y - yp
		if self.bias_correction is not None:
			diff += self.bias_correction
		return diff

	def add_batch(self, batch, result):
		y = batch[self.target]
		
		if self.model_output is None:
			yp = result
		else:
			if type(self.model_output) is list:
				for idx in self.model_output:
					result = result[idx]
			else:
				result = result[self.model_output]
			yp = result

		diff = self._get_diff(y, yp)
		if self.atom_wise:
			diff /= torch.sum(batch[Properties.atom_mask], dim=1, keepdim=True)
		
		self.l2loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
		if self.element_wise:
			self.n_entries += (
				torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
				* y.shape[-1]
			)
		else:
			self.n_entries += np.prod(y.shape)

	def aggregate(self):
		return self.l2loss / self.n_entries


class RootMeanSquaredError(MeanSquaredError):
	r"""
	Metric for root mean square error. For non-scalar quantities, the mean of
	all components is taken.
	Args:
		target (str): name of target property
		model_output (int, str): index or key, in case of multiple outputs
			(Default: None)
		name (str): name used in logging for this metric. If set to `None`,
			`RMSE_[target]` will be used (Default: None)
		element_wise (bool): set to True if the model output is an element-wise
			property (forces, positions, ...)
		atom_wise (bool): set to True to divide the model output by the number
			of atoms to get eV per atom, etc.
	"""

	def __init__(
		self,
		target,
		model_output=None,
		bias_correction=None,
		name=None,
		element_wise=False,
		atom_wise=False
	):
		name = "RMSE_" + target if name is None else name
		super(RootMeanSquaredError, self).__init__(
			target, model_output, bias_correction, name, element_wise=element_wise, atom_wise=atom_wise
		)

	def aggregate(self):
		"""Aggregate metric over all previously added batches."""
		return np.sqrt(self.l2loss / self.n_entries)

class MeanAbsoluteError(Metric):
	r"""
	Metric for mean absolute error. For non-scalar quantities, the mean of all
	components is taken.
	Args:
		target (str): name of target property
		model_output (int, str): index or key, in case of multiple outputs
			(Default: None)
		name (str): name used in logging for this metric. If set to `None`,
			`MAE_[target]` will be used (Default: None)
		element_wise (bool): set to True if the model output is an element-wise
			property (forces, positions, ...)
		atom_wise (bool): set to True to divide the model output by the number
			of atoms to get eV per atom, etc.
	"""

	def __init__(
		self,
		target,
		model_output=None,
		bias_correction=None,
		name=None,
		element_wise=False,
		atom_wise=False
	):
		name = "MAE_" + target if name is None else name
		super(MeanAbsoluteError, self).__init__(
			target=target,
			model_output=model_output,
			name=name,
			element_wise=element_wise,
		)

		self.bias_correction = bias_correction
		self.atom_wise = atom_wise

		self.l1loss = 0.0
		self.n_entries = 0.0

	def reset(self):
		"""Reset metric attributes after aggregation to collect new batches."""
		self.l1loss = 0.0
		self.n_entries = 0.0

	def _get_diff(self, y, yp):
		diff = y - yp
		if self.bias_correction is not None:
			diff += self.bias_correction
		return diff

	def add_batch(self, batch, result):
		y = batch[self.target]

		if self.model_output is None:
			yp = result
		else:
			if type(self.model_output) is list:
				for idx in self.model_output:
					result = result[idx]
					# print(result.shape)
			else:
				result = result[self.model_output]
			yp = result

		diff = self._get_diff(y, yp)
		if self.atom_wise:
			diff /= torch.sum(batch[Properties.atom_mask], dim=1, keepdim=True)
		
		self.l1loss += (
			torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
		)
		if self.element_wise:
			self.n_entries += (
				torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
				* y.shape[-1]
			)
		else:
			self.n_entries += np.prod(y.shape)

	def aggregate(self):
		"""Aggregate metric over all previously added batches."""
		return self.l1loss / self.n_entries

class R2Score(Metric):
	r"""
	Metric for R square score.

	Args:
		target (str): name of target property
		model_output (int, str): index or key, in case of multiple outputs
			(Default: None)
		name (str): name used in logging for this metric. If set to `None`,
			`MSE_[target]` will be used (Default: None)
		element_wise (bool): set to True if the model output is an element-wise
			property (forces, positions, ...)
		atom_wise (bool): set to True to divide the model output by the number
			of atoms to get eV per atom, etc.
	"""

	def __init__(
		self,
		target,
		model_output=None,
		bias_correction=None,
		name=None,
		element_wise=False,
		atom_wise=False
	):
		name = "R2_" + target if name is None else name
		super(R2Score, self).__init__(
			target=target,
			model_output=model_output,
			name=name,
			element_wise=element_wise,
		)

		self.bias_correction = bias_correction
		self.atom_wise = atom_wise

		self.y_sum = 0.0
		self.y_sq_sum = 0.0
		self.l2loss = 0.0
		self.n_entries = 0.0

	def reset(self):
		"""Reset metric attributes after aggregation to collect new batches."""
		self.y_sum = 0.0
		self.y_sq_sum = 0.0
		self.l2loss = 0.0
		self.n_entries = 0.0

	def _get_diff(self, y, yp):
		diff = y - yp
		if self.bias_correction is not None:
			diff += self.bias_correction
		return diff

	def add_batch(self, batch, result):
		y = batch[self.target]
		
		if self.model_output is None:
			yp = result
		else:
			if type(self.model_output) is list:
				for idx in self.model_output:
					result = result[idx]
			else:
				result = result[self.model_output]
			yp = result

		diff = self._get_diff(y, yp)
		if self.atom_wise:
			diff /= torch.sum(batch[Properties.atom_mask], dim=1, keepdim=True)
		
		self.y_sum += torch.sum(y.view(-1)).detach().cpu().data.numpy()
		self.y_sq_sum += torch.sum(y.view(-1) ** 2).detach().cpu().data.numpy()
		self.l2loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()

		if self.element_wise:
			self.n_entries += (
				torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
				* y.shape[-1]
			)
		else:
			self.n_entries += np.prod(y.shape)

	def aggregate(self):
		return 1 - self.l2loss / (self.y_sq_sum - (self.y_sum ** 2) / self.n_entries)
