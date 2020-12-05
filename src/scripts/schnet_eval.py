import os
import torch
#import numpy as np

#from sys import argv
from natsort import natsorted

from schnetpack.data import AtomsData, AtomsLoader
from schnetpack.utils import load_model, evaluate_dataset
from schnetpack.train import MeanAbsoluteError, RootMeanSquaredError, CSVHook

from schnetpack_tools.metrics import R2Score
from schnetpack_tools.environment import OpenCLEnvironmentProvider

def format_results(dataset_file, model_file, results):

	output = '%20s %20s' % (dataset_file, model_file)
	output += len(results) * ' %11.7f' % tuple(results)
	output += '\n'

	return output

device = torch.device('cuda')
cutoff = 5.0
properties = ['energy', 'forces', 'stress']

models_path = '../models/'
datasets_path = 'datasets/'

metrics = [
	MeanAbsoluteError('energy', 'energy'),
	MeanAbsoluteError('forces', 'forces', element_wise=True),
	MeanAbsoluteError('stress', 'stress'),
	RootMeanSquaredError('energy', 'energy'),
	RootMeanSquaredError('forces', 'forces', element_wise=True),
	RootMeanSquaredError('stress', 'stress'),
	R2Score('energy', 'energy'),
	R2Score('forces', 'forces', element_wise=True),
	R2Score('stress', 'stress')]

fid = open('results.txt', 'w')

header  = '             Dataset                Model  MAE energy  MAE forces  MAE stress RMSE energy RMSE forces RMSE stress   R2 energy   R2 forces   R2 stress\n'
header += '==================== ==================== =========== =========== =========== =========== =========== =========== =========== =========== ===========\n'

fid.write(header)
fid.flush()

for dataset_file in natsorted(os.listdir(datasets_path)):

	dataset = AtomsData(datasets_path + dataset_file, load_only=properties,
		environment_provider=OpenCLEnvironmentProvider(cutoff, 0),
		centering_function=None)

	loader = AtomsLoader(dataset, batch_size=20, num_workers=1, pin_memory=True)

	for model_file in natsorted(os.listdir(models_path)):

		model = load_model(models_path + model_file)

		# Disable the creation of graph, which is not needed since we are only evaluating.
		model.output_modules[0].create_graph = False

		fid.write(format_results(dataset_file, model_file, evaluate_dataset(metrics, model, loader, device)))
		fid.flush()

fid.close()
