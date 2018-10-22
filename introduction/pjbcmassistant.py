#this package contains convenience functions for handling
#the business of using Python 3.7 Jupyter Notebooks and
#the Python package pyjags, along with JAGS, to
#perform the code exercises in the textbook 
#Bayesian cognitive modeling: A practical course,
#by Lee, M. D., & Wagenmakers, E. J. (2014)

from os import path
#import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

class model_spec:
	"""
	Handle loading, storing, providing, and displaying files that specify
	a JAGS model.
	"""
	def __init__(self, model:str ='', data:str ='', init:str ='') -> None:
		self.dir = path.dirname(path.realpath(__file__))
		self.load(model, data, init)

	def load_model(self, filepath:str) -> None:
		with open(path.join(self.dir +'/'+ filepath), 'r') as modelfile: #this should not need that slash...
			model_str = modelfile.read()
		return model_str

	def load_data(self, filepath:str) -> None:
		data_str = ''
		with open(filepath, 'r') as datafile:
			data_str += [line for line in datafile]
		return data_str

	def load_init(self, filepath:str) -> None:
		# with open(path.join(self.dir +'/'+ filepath), 'r') as initfile: #this should not need that slash...
		# 	init_dict = {}
		# 	init_str = initfile.read()
		# 	pattern = r'[\S ]*=[\S ]*' #catches this = that
		# 	matches = re.search(pattern, init_str)
		# 	for match in matches.groups():
		# 		args = match.replace(' ','').split('=')
		# 		init_dict[args[0]] = init

		# 	chainvals = init_str.split("")
		# return init_str
		pass

	def load(self, model:str ='', data:str ='', init:str ='') -> None:
		if model: self.model = self.load_model(model)
		if data: self.data = self.load_data(data)
		if init: self.init = self.load_init(init)

	# def set(self, attribute, value) -> None:
	# 	attribute = value

	def getvars(self) -> str:
		#it'd be nice to grab and report all the variables used in the model
		pass

class visualizer:
	"""
	Handle common display functions for data in BCM exercises.
	"""
	def __init__(self, samples) -> None:
		data = {k: v.squeeze(0) for k, v in samples.items()}
		nsamples = len(next(iter(data.values())))
		myvars = [k for k in data.keys()]
		nchains = len(next(iter(data.values()))[0])

		idx=pd.MultiIndex.from_product([[i for i in range(nchains)],[i for i in range(nsamples)]])
		idx.set_names(['chain', 'sample'], inplace=True)

		

		self.df = pd.DataFrame(
			index=idx,
			columns=[i for i in myvars]
		)

		self.df.rename_axis(["variable"], axis=1, copy=False, inplace=True) #name columns

		for key in data:
			for chain in range(nchains):
				self.df[key][chain] = data[key][:,chain]

		self.df = self.df.astype(float)

		
		
	def head(self):
		print(self.df.head())

	def load(self, dataframe) -> None:
		self.df = dataframe

	def chains(self, *variables, range=100):

		fig, ax = plt.subplots(figsize=(8,3))
		

		#self.df[variable].groupby('chain')[0:range].plot(ax=ax)
		self.df.groupby('chain')[[i for i in variables]].apply(lambda x: x).unstack(level=0)[:range].plot(ax=ax)

		plt.ylabel("parameter value")
		plt.title(f"First {range} samples of {[i for i in(variables)]}")

		
		plt.show()

		# self.df[[i for i in variables]].hist()


	def hist(self, *variables, bins=50, range=(0,1)):
		
		fig, ax = plt.subplots(figsize=(6,6))
		if len(variables)>1:
			self.df.hist([i for i in variables], range=range, bins=bins)
		else:
			self.df.hist([i for i in variables], range=range, bins=bins, ax=ax)

		#plt.hist(self.df[variable].values.tolist(),range=range, bins=bins)
		plt.ylabel("frequency")
		plt.xlabel("parameter value")
		plt.title(f"Frequency of sampled values of {[i for i in variables]}")
		plt.show()


	def kde(self, variable, showmax=False):
		#use scipy to generate KDE function from a numpy array pulled from the dataframe
		nparam_density = stats.gaussian_kde(self.df[variable].values.ravel())

		#generate a linear space for the function to be calculated over
		x = np.linspace(0.001,1,1000)

		#plot x against kde(x)
		fig, ax = plt.subplots(figsize=(5,5))
		ax.plot(x, nparam_density(x), label='non-parametric density (smoothed by Gaussian kernel)')

		plt.ylabel("density")
		plt.xlabel(f"{variable}")
		plt.title(f"Kernel Density Estimate for {variable}")

		plt.show()

		if showmax:
			max_density = x[np.argsort(nparam_density(x))[-1]]
			print(f"maximum density observed across 1000 bins was at x = {max_density}")















