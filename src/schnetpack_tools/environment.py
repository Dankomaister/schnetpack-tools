import os
import numpy as np
import pyopencl as cl

from schnetpack.environment import BaseEnvironmentProvider

class OpenCLEnvironmentProvider(BaseEnvironmentProvider):
	"""docstring for OpenCLEnvironmentProvider"""

	def __init__(self, cutoff, platform=0, number_density=0.30):
		"""
		Args:
			cutoff (float): the cutoff inside which atoms are considered pairs.
			platform (int): set which  opencl device to use.
			number_density (float): sets the maximum number of neighbours per atom.
									as np.ceil(4/3*np.pi*cutoff**3 * number_density)
		"""
		self.cutoff = np.float64(cutoff)
		self.max_nbh = np.uint32(np.ceil(4/3*np.pi*cutoff**3 * number_density))

		self.kernel_code = ''.join(open(os.path.dirname(os.path.realpath(__file__)) + '/neighbor_list_kernel.cl', 'r').readlines())
		self.kernel = None
		self.platform = platform
		self.mf = cl.mem_flags

	def initialize(self):
		devices = cl.get_platforms()[self.platform].get_devices()
		self.ctx = cl.Context(devices)
		self.queue = cl.CommandQueue(self.ctx)
		self.kernel = cl.Program(self.ctx, self.kernel_code).build()
		print('OpenCLEnvironmentProvider initialized on: %s' % devices[0])

	def get_environment(self, atoms):
		if self.kernel is None:
			self.initialize()

		n_atoms = np.uint32(len(atoms))
		cell = np.array(atoms.cell, order='C')
		positions = np.array(atoms.positions, order='C')
		scaled_positions = np.zeros((n_atoms,4), dtype=np.float64, order='C')
		scaled_positions[:,0:3] = np.linalg.solve(cell.T, positions.T).T.astype(np.float64)

		neighborhood_idx = -np.ones((n_atoms,self.max_nbh), dtype=np.int32, order='C')
		offset = np.zeros((n_atoms,self.max_nbh), dtype=cl.cltypes.short3, order='C') # Change to integer to improve perf.

		scaled_positions_buffer = cl.Buffer(self.ctx, self.mf.READ_ONLY  | self.mf.COPY_HOST_PTR, hostbuf=scaled_positions)
		neighborhood_idx_buffer = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=neighborhood_idx)
		offset_buffer           = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=offset)

		cell = cell.astype(np.float64)
		self.kernel.neighbor_list(self.queue, (n_atoms,), None, scaled_positions_buffer, offset_buffer, neighborhood_idx_buffer,
			n_atoms, self.max_nbh, self.cutoff,
			cell[0,0], cell[0,1], cell[0,2],
			cell[1,0], cell[1,1], cell[1,2],
			cell[2,0], cell[2,1], cell[2,2])

		e1 = cl.enqueue_copy(self.queue, offset, offset_buffer)
		e2 = cl.enqueue_copy(self.queue, neighborhood_idx, neighborhood_idx_buffer)

		e2.wait()
		tmp = neighborhood_idx.copy()
		tmp[tmp != -1] = 0
		I = np.uint32(self.max_nbh + np.max(np.sum(tmp,1)))
		if I == 0:
			I = 1

		offset_out = np.zeros((n_atoms,I,3), dtype=np.float64, order='C')
		e1.wait()
		offset_out[:,:,0] = offset[:,0:I]['x']
		offset_out[:,:,1] = offset[:,0:I]['y']
		offset_out[:,:,2] = offset[:,0:I]['z']

		return neighborhood_idx[:,0:I], offset_out
