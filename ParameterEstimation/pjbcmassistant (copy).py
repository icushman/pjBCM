#this package contains convenience functions for handling
#the business of using Python 3.7 Jupyter Notebooks and
#the Python package pyjags, along with JAGS, to
#perform the code exercises in the textbook 
#Bayesian cognitive modeling: A practical course,
#by Lee, M. D., & Wagenmakers, E. J. (2014)

from os import path
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

class model_handler:
	"""
	Handle loading, storing, providing, and displaying files that specify
	a JAGS model.
	"""
	def __init__(self, settings) -> None:
		self.configure(settings)
		self.model = self.load_model(self.modelfile)
		

	def load_model(self, filepath:str) -> None:
		mydir = path.dirname(path.realpath(__file__))
		with open(path.join(mydir +'/'+ filepath), 'r') as modelfile: #this should not need that slash...
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

	def configure(self, spec):
		#got to be a better way to do this... if exists testing...
		self.init = None
		self.chains = 4
		self.burnin = 1000
		self.thinning = 0
		
		self.data = {}
		self.sample = []
		section = ''
		for line in spec.split('\n'):
			if ':' in line:
				section = line

			elif len(line) > 0:


				if 'model' in section:
					#this could use an allowance for loading the text directly? yes.
					self.modelfile = line
				elif section == 'settings:':
					assignment = line.replace(" ","").split('=')
					self.__dict__[assignment[0]] = assignment[1]
				elif section == 'data:':
					# do we EVER pass non-numeric data to models?
					assignment = line.replace(" ","").split('=')
					if assignment[1].startswith('['):
						#should this be more clever about deciding to int or float the values?
						self.data[assignment[0]] = np.array([float(i) for i in assignment[1].strip("[]").split(',')])
					else:
						self.data[assignment[0]] = assignment[1]
				elif section == 'sample:':
					self.sample.append(line)
				

	def definition(self):
		definition = {'file':self.modelfile,
			#'init':self.init,
			'data':self.data if len(self.data) > 0 else None,
			'chains':int(self.chains),
			'adapt':int(self.burnin),
		}
		return definition


	def sample_rules(self):
		definition = {'vars':self.sample,
			'iterations':int(self.samples),
		}
		if int(self.thinning) > 0:
			definition['thin'] = int(self.thinning)
		return definition




	# def __setattr__(self, index, value):
	# 	self.__dict__[index] = value


#grab data contents : (?:data:\s)([\s\S]*)\s\s

class sample_handler:
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

		

		self.samples = pd.DataFrame(
			index=idx,
			columns=[i for i in myvars]
		)

		self.samples.rename_axis(["variable"], axis=1, copy=False, inplace=True) #name columns

		for key in data:
			for chain in range(nchains):
				self.samples[key][chain] = data[key][:,chain]

		self.samples = self.samples.astype(float)


	def get(self, variable):
		return self.samples[variable].values
		
		
	def head(self):
		print(self.samples.head())

	def load(self, dataframe) -> None:
		self.samples = dataframe

	def vizchains(self, *variables, range=100):

		fig, ax = plt.subplots(figsize=(8,3))
		

		#self.samples[variable].groupby('chain')[0:range].plot(ax=ax)
		self.samples.groupby('chain')[[i for i in variables]].apply(lambda x: x).unstack(level=0)[:range].plot(ax=ax)

		plt.ylabel("parameter value")
		plt.title(f"First {range} samples of {[i for i in(variables)]}")

		
		plt.show()

		# self.samples[[i for i in variables]].hist()


	def vizhist(self, *variables, bins=50, range=(0,1)):
		
		
		if len(variables)>1:
			self.samples.hist([i for i in variables], range=range, bins=bins)
		else:
			fig, ax = plt.subplots(figsize=(6,6))
			self.samples.hist([i for i in variables], range=range, bins=bins, ax=ax)

		#plt.hist(self.samples[variable].values.tolist(),range=range, bins=bins)
		plt.ylabel("frequency")
		plt.xlabel("parameter value")
		plt.title(f"Frequency of sampled values of {[i for i in variables]}")
		plt.show()


	def vizkde(self, variable, showmax=False, range=(0.001,1)):
		#use scipy to generate KDE function from a numpy array pulled from the dataframe
		nparam_density = stats.gaussian_kde(self.samples[variable].values.ravel())

		#generate a linear space for the function to be calculated over
		x = np.linspace(range[0],range[1],1000)

		#plot x against kde(x)
		fig, ax = plt.subplots(figsize=(5,5))
		ax.plot(x, nparam_density(x), label='non-parametric density (smoothed by Gaussian kernel)')

		plt.ylabel("density")
		plt.xlabel(f"{variable}")
		plt.title(f"Kernel Density Estimate for {variable}")
		plt.xlim=range

		plt.show()

		if showmax:
			max_density = x[np.argsort(nparam_density(x))[-1]]
			print(f"maximum density observed across 1000 bins was at x = {max_density}")


	def summarize(self, variable, ci="95"):
		#should be edited to support arbitrary joint reports?
		var = self.get(variable)
		samples = len(var)

		mean = np.mean(var)
		median = np.median(var)
		
		cip = self.ci_percentile(var)
		cihpd = self.ci_hpd(var)
		mode = self.mode(var)
		
		return str(f'variable:{variable}\nmean:{mean}\nmedian:{median}\nmode:{mode}\n2.5th-97.5th %ile:{cip}\n95% HPD:{cihpd}')

	def ci_percentile(self,variable, bounds=(2.5,97.5)):
		return np.percentile(variable,bounds[0]),np.percentile(variable,bounds[1],interpolation='nearest')

	def ci_hpd(self,var, bins=1000, interval=95):
		
		#this is built for unimodal data, eww
		#it's also VERY approximate due to its formulation.
		#better to copy the one in intro, I think.

		#however, this one can (with a little tweaking) deal with
		#multimodal data, which that one cannot.

		#there's got to be a RIGHT way to do this, i'm sure.

		#DRY lol
		hist = np.histogram(var,bins)

		binvals = hist[0]
		bins = hist[1]

		values_to_capture = (interval/100)*len(var)

		bintuples = [(bins[i],bins[i+1]) for i in range(len(bins)-1)]

		binpairs = zip(binvals, bintuples)

		sortedlist = sorted(binpairs, key=(lambda x : x[0]), reverse=True)

		#print([i for i in sortedlist])

		values_captured = 0
		bins_included = []
		for histbin in sortedlist:
			values_captured += histbin[0]
			bins_included.append(histbin[1])
			if values_captured >= values_to_capture:
				break
				
				
		#make this test smarter to catch multiple disjoint CIs?
		span_captured = (min([i[0] for i in bins_included]), max([i[1] for i in bins_included]))
				
		#print(values_captured, bins_included)

		return span_captured
		
		


		return maxbin

		return np.percentile(variable,bounds[0]),np.percentile(variable,bounds[1],)

	def mode(self, variable, bins=100):
		bins = min([bins, len(variable)//5]) #so there's always a decent bincount
		hist = np.histogram(variable,bins)
		maxindex = np.argmin(hist[0])
		maxbin = np.mean([hist[1][maxindex],hist[1][maxindex+1]])
		return maxbin
			












