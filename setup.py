from setuptools import setup, find_packages

setup(
	name = 'schnetpack_tools',
	version = '0.1',
	author = 'Daniel Hedman',
	author_email = 'daniel.hedman@ltu.se',
	url = 'https://github.com/Dankomaister/schnetpack-tools',
	packages = find_packages('src'),
	package_dir = {'': 'src'},
	package_data={'': ['neighbor_list_kernel.cl']},
	include_package_data = True,
	install_requires = [
		'pyopencl',
		'numpy',
		'schnetpack'
	],
	license = 'MIT',
	description='Additional tools for SchNetPack.'
)
