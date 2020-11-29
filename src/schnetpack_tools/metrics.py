import torch
import numpy as np

from schnetpack import Properties
from schnetpack.train import Metric

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
	"""

	def __init__(
		self,
		target,
		model_output=None,
		bias_correction=None,
		name=None,
		element_wise=False,
	):
		name = "R2_" + target if name is None else name
		super(R2Score, self).__init__(
			target=target,
			model_output=model_output,
			name=name,
			element_wise=element_wise,
		)

		self.bias_correction = bias_correction

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
