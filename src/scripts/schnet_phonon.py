import os
import torch
import phonopy
import numpy as np

#from sys import argv
from natsort import natsorted

from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from ase.io.vasp import write_vasp
from ase.constraints import ExpCellFilter

from schnetpack.utils import load_model
from schnetpack.interfaces import SpkCalculator
from schnetpack_tools.environment import OpenCLEnvironmentProvider

from phonopy.phonon.band_structure import get_band_qpoints

import matplotlib.pyplot as plt

###### Settings for matplotlib ######
from matplotlib import rcParams

def cm(x):
	return 0.393700787*x

ratio = 0.90
fig_width = cm(8.0)
fig_height = ratio*fig_width

rcParams['figure.dpi'] = 300
rcParams['figure.figsize'] = [fig_width, fig_height]

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'DejaVu Serif'
rcParams['mathtext.fontset'] = 'dejavuserif'

rcParams['font.size'] = 8
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 6

rcParams['lines.markersize'] = 1.0
rcParams['lines.linewidth'] = 0.75
rcParams["legend.handlelength"] = 1.0

#####################################

def relax_poscar(poscar_file, calc, fmax=1e-4):
	atoms = read(poscar_file)
	atoms.set_calculator(calc)

	ecf = ExpCellFilter(atoms)

	opt = BFGS(ecf)
	opt.run(fmax=fmax)

	write_vasp('POSCAR-opt', atoms, vasp5=True)

	return

def spk_calculator(model_file, environment_provider, device):

	model = load_model(model_file)
	model.output_modules[0].create_graph = False

	return SpkCalculator(model=model, device=device, environment_provider=environment_provider,
		energy='energy', forces='forces', stress='stress')

def calculate_forces(phonon, calc, correct_drift=True):
	supercells = phonon.get_supercells_with_displacements()
	force_list = []

	for supercell in supercells:
		atoms = Atoms(
			symbols=supercell.get_chemical_symbols(),
			cell=supercell.get_cell(),
			scaled_positions=supercell.get_scaled_positions(),
			pbc=True)

		atoms.set_calculator(calc)
		forces = atoms.get_forces()

		if correct_drift:
			drift = np.average(forces, axis=0)
			forces -= drift
			print('Drift of force constants: %9.6f %9.6f %9.6f' % tuple(drift))

		force_list.append(forces)

	return force_list

def plot_band_structure(spk, ref, band_path, labels, fc2_mae, name):

	spk_band = spk.get_band_structure_dict()
	ref_band = ref.get_band_structure_dict()
	spk_freqs = np.vstack(spk_band['frequencies'])
	ref_freqs = np.vstack(ref_band['frequencies'])
	qpoints = np.vstack(ref_band['qpoints'])
	distances = np.hstack(ref_band['distances'])

	ticks = np.unique(np.hstack([distances[np.all(qpoints == qp, axis=1)] for qp in np.vstack(band_path)]))

	plt.figure(1)
	plt.plot(distances, ref_freqs, 'k')
	plt.plot(distances, spk_freqs, 'tab:red')
	plt.hlines(0, 0, distances[-1], linestyle=':', color='0.69')
	plt.grid(axis='x')

	plt.xlim(0, distances[-1])
	plt.ylim(-1, 25)

	plt.xticks(ticks, labels)

	plt.text(0.05, 0.95, r'MAE $\Phi_{\alpha\beta}: %.4g$ meV/Å$^2$' %
		(1e3*fc2_mae), transform=plt.gca().transAxes, fontsize=rcParams['legend.fontsize'])

	plt.ylabel('Frequency (THz)')

	plt.tight_layout(pad=0.1)
	plt.savefig('results/' + name + '.png', pad_inches=0.0)
	#plt.savefig('MC_%s.pdf' % model.split('_')[-2], pad_inches=0.0)
	plt.close()

	return


def fc2_mae(spk, ref):
	return np.mean(np.abs(spk.get_force_constants() - ref.get_force_constants()))

def freq_mae(spk, ref, band_path):
	return np.mean([np.abs(spk.get_frequencies(hsp) - ref.get_frequencies(hsp)) for hsp in band_path])

def bands_mae(spk, ref, bands):
	spk.run_band_structure(bands)
	spk_bands = np.array(spk.get_band_structure_dict()['frequencies'])
	ref_bands = np.array(ref.get_band_structure_dict()['frequencies'])

	return np.mean(np.abs(spk_bands - ref_bands))

### Settings ###

models_path = '../../models/'
device = torch.device('cuda')
environment_provider = OpenCLEnvironmentProvider(5.0, 0)

relax = False
supercell = (4,4,4)
systems = ['HfC', 'NbC', 'TaC', 'TiC', 'ZrC']

labels = [r'$\Gamma$', 'X', 'U,K', r'$\Gamma$', 'L', 'W', 'X']
band_path = [
	[[0.0000000000, 0.0000000000, 0.0000000000],
	 [0.5000000000, 0.0000000000, 0.5000000000],
	 [0.6250000000, 0.2500000000, 0.6250000000]],
	[[0.3750000000, 0.3750000000, 0.7500000000],
	 [0.0000000000, 0.0000000000, 0.0000000000],
	 [0.5000000000, 0.5000000000, 0.5000000000],
	 [0.5000000000, 0.2500000000, 0.7500000000],
	 [0.5000000000, 0.0000000000, 0.5000000000]]]

################

fid = open('results.txt', 'w')
fid.write('              System                Model     MAE fc2   MAE freq.   MAE bands\n')
fid.write('==================== ==================== =========== =========== ===========\n')

for system in systems:

	phonon_ref = phonopy.load(
		unitcell_filename=system+'/POSCAR-unitcell',
		supercell_matrix=supercell,
		force_constants_filename=system+'/FORCE_CONSTANTS')

	bands = get_band_qpoints(band_path, 100)
	phonon_ref.run_band_structure(bands)

	for model_file in natsorted(os.listdir(models_path)):

		calc = spk_calculator(models_path+model_file, environment_provider, device)

		if relax:
			relax_poscar(system+'/POSCAR-unitcell', calc)
			phonon = phonopy.load(
				unitcell_filename='POSCAR-opt',
				supercell_matrix=supercell,
				produce_fc=False)
		else:
			phonon = phonopy.load(
				unitcell_filename=system+'/POSCAR-unitcell',
				supercell_matrix=supercell,
				produce_fc=False)

		phonon.generate_displacements(distance=0.03)
		phonon.produce_force_constants(forces=calculate_forces(phonon, calc))
		phonon.run_band_structure(bands)

		plot_band_structure(phonon, phonon_ref, band_path, labels, 0.0, '%s-%s' % (system,model_file))

		fid.write('%20s %20s %11.7f %11.7f %11.7f\n' %
			(system, model_file, fc2_mae(phonon, phonon_ref), freq_mae(phonon, phonon_ref, band_path), bands_mae(phonon, phonon_ref, bands)))
		fid.flush()

fid.close()
