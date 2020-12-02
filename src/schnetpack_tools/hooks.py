from schnetpack.train.hooks import Hook
from torch.optim.lr_scheduler import StepLR

class FindBestLearningRateHook(Hook):
	r"""
	Hook for finding a good learning rate.
	Results are written to a find_lr.csv file.
	
	Args:
		final_lr (float): The final learning rate.
		steps (int): Total number of learning rate steps.
	"""

	def __init__(self, optimizer, final_lr=1e-3, steps=100):

		lr = optimizer.param_groups[0]['lr']

		self.fid = open('find_lr.csv', 'w')

		self.i = 1
		self.steps = steps
		self.scheduler = StepLR(optimizer, 1, (final_lr/lr)**(1/(steps-1)))

	def on_batch_end(self, trainer, train_batch, result, loss):
		if self.i > self.steps:
			trainer._stop = True
			self.fid.close()
			return

		line  = str(self.i)
		line += str(',')
		line += str(trainer.optimizer.param_groups[0]['lr'])
		line += str(',')
		line += str(loss.data.cpu().numpy())
		line += str('\n')

		#self.fid.write('%3i,%12.5e,%10.5f\n' % (self.i, trainer.optimizer.param_groups[0]['lr'], loss.data.cpu().numpy()))
		self.fid.write(line)
		self.fid.flush()

		self.scheduler.step()
		self.i += 1
