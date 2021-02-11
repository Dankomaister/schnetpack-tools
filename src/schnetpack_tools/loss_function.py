import torch

class R2Loss(object):
	"""docstring for R2Loss"""
	def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):

		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

	def r2_err(self, y, yp):
		return torch.sum((y - yp) ** 2) / torch.sum((y - torch.mean(y)) ** 2)

	def get_loss(self, batch, result):
		with torch.no_grad():
			# Index for non-padded forces
			I = batch['_atom_mask'].bool()

		E_err = self.alpha * self.r2_err(batch['energy'], result['energy'])
		F_err = self.beta  * self.r2_err(batch['forces'][I], result['forces'][I])
		S_err = self.gamma * self.r2_err(batch['stress'], result['stress'])

		loss = 0.0

		if not torch.isnan(E_err) and not torch.isinf(E_err):
			loss += E_err

		if not torch.isnan(F_err) and not torch.isinf(F_err):
			loss += F_err

		if not torch.isnan(S_err) and not torch.isinf(S_err):
			loss += S_err

		return loss

