#this package contains convenience functions for handling
#the business of using Python 3.7 Jupyter Notebooks and
#the Python package pyjags, along with JAGS, to
#perform the code exercises in the textbook 
#Bayesian cognitive modeling: A practical course,
#by Lee, M. D., & Wagenmakers, E. J. (2014)

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

class model_handler:
	"""Handle operations on data that specifies a JAGS model."""

	def __init__(self, spec:str) -> None:
		"""Read in model spec string and bind needed values"""
		self.configure(spec)
		self.model = self.load_model(self.modelfile)
		
	def load_model(self, filename:str) -> str:
		"""Load model file specified in spec string."""
		#TODO: handle model being specified as sting inline in notebooks.
		with open(os.path.join(os.getcwd(), filename), 'r') as modelfile:
			return modelfile.read()

	def configure(self, spec):
		"""Read spec string and bind all values needed for pyjags.Model"""
		#TODO: Refactor this completely. It's nice to have the model spec look like
		#a simple settings file, but it's adding a lot of brittle complexity
		#try to see who the users are, and if this makes sense for them.
		#If so, rewrite it -- there are conventions for this kind of parsing,
		#and we're not following them.

		#initialize arguments to be passed to pyjags.Model
		self.init = None
		self.samples = 1000
		self.chains = 4
		self.burnin = 1000
		self.thinning = 0
		self.data = {}
		self.sample = []

		#Parse spec string
		#TODO: Create set method to adjust these settings outside the spec string
		section = ''
		for settingsline in spec.split('\n'):
			#throw out comments
			if "#" in settingsline:
				settingsline = settingsline.split("#")[0]

			#detect section headers	
			if ':' in settingsline:
				section = settingsline

			elif len(settingsline) > 0:
				if section == 'model:': #'model' in section:
					self.modelfile = settingsline
				elif section == 'settings:':
					assignment = settingsline.replace(" ","").split('=')
					self.__dict__[assignment[0]] = assignment[1]
				elif section == 'data:':
					#break line into variable names and values
					assignment = settingsline.replace(" ","").split('=')
					expression = assignment[1]
					variables = assignment[0].split(',')
					#use regex to replace model variable names with names
					#appropriate for storage in the model.
					#This is NOT the clever way to do this, if it even should happen.
					for key in self.data.keys():
						expression = re.sub(fr'(\s{key}$|^{key}(?:\s|$)|\s{key}\s|\({key}\)|\[{key}\])',lambda match : (match.group(0).replace(str(key),f'self.data["{key}"]')), expression)
					#handle multiple variable assignment
					if len(variables) > 1:
						for index, variable in enumerate(variables):
							self.data[variable] = eval(expression)[index]
					#handle single variable assignment
					else:
						self.data[variables[0]] = eval(expression)
				elif section == 'sample:':
					self.sample.append(settingsline)
				

	def get_model_args(self):
		"""Create keyword arguments dictionary for pyjags.model.Model"""
		model_args = {'file':self.modelfile,
			'init':self.init,
			'data':self.data if len(self.data) > 0 else None,
			'chains':int(self.chains),
			'adapt':int(self.burnin),
		}
		return model_args

	def get_sample_args(self):
		"""Create keyword arguments dictionary for pyjags.model.Model.sample"""
		sample_args = {'vars':self.sample,
			'iterations':int(self.samples),
		}
		if int(self.thinning) > 0:
			sample_args['thin'] = int(self.thinning)
		return sample_args

	def setdata(self, varname, value) -> None:
		"""Method to set model's data values outside of spec string."""
		self.data[varname] = value

class sample_handler:
	"""Handle data processing and display functions for common textbook exercises."""

	def __init__(self, samples) -> None:
		"""Format and label samples returned by pyjags.model.Model.sample"""
		#TODO: Refactor this function to be more readable
		#####  We shouldn't need this many comments.

		#Build dictionary of sample values
		data = {}
		for k, v in samples.items():
			if np.shape(samples[k])[0] == 1:
			#if parameter has only one dimension...
				data[k] = v.squeeze(0)
				#... ensure that it is stored as a 1D series.
			else:
			#if parameter has multiple dimension (e.g., mu_0, mu_1...)
				for i in range(np.shape(samples[k])[0]):
					data[f"{k}_{str(i)}"] = v[i,:,:]
					#number each sub-parameter and give it
					#its own entry in our data dictionary

		#extract nsamples, nchains, and tracked parameter names from data.
		nsamples = len(next(iter(data.values())))
		nchains = len(next(iter(data.values()))[0])
		self.parameters = [k for k in data.keys()]

		#we will store data in pandas MultiIndex dataframe
		#with dimensions chain, sample number, and parameter
		#this builds the frame's indices:  
		idx=pd.MultiIndex.from_product([[i for i in range(nchains)],[i for i in range(nsamples)]])
		idx.set_names(['chain', 'sample'], inplace=True)
		#construct 3D dateframe
		self.samples = pd.DataFrame(
			index=idx,
			columns=[i for i in self.parameters]
		)
		self.samples.rename_axis(["variable"], axis=1, copy=False, inplace=True) #name columns

		try:
			#load values from data dict into self.samples DataFrame
			for key in data:
				for chain in range(nchains):
					self.samples[key][chain] = data[key][:,chain]
		except IndexError:
			assert False, (f'Debugging note: error likely relates to unpacking nested vars')

		self.samples = self.samples.astype(float)

		self.variable_statistics = {}
		

	def get(self, variable):
		"""Return all sample values of given variable"""
		return self.samples[variable].values
		
		
	def head(self):
		"""Print first 5 rows of samples dataframe"""
		print(self.samples.head())


	def vizchains(self, *variables, range=100):
		"""Generate line plot for the first [range] samples of given variable(s)"""
		fig, ax = plt.subplots(figsize=(8,3))
		self.samples.groupby('chain')[[i for i in variables]].apply(lambda x: x).unstack(level=0)[:range].plot(ax=ax)
		plt.ylabel("parameter value")
		plt.title(f"First {range} samples of {[i for i in(variables)]}")
		plt.show()

	def vizjoint(self, variable1, variable2):
		"""Produce joint plot of given variables."""
		joint = sns.jointplot(variable1, variable2, data=self.samples, color="grey");

	def vizhist(self, *variables, bins=50, range:(int,int)=None):
		"""Produce histogram of given variable(s)"""
		#TODO: Make layout of subplots more usable, clean subplot code.
		if range == None:
			#choose x limits of axes automatically if not specified.
			if len(variables)>1:
				self.samples.hist([i for i in variables], bins=bins)
			else:
				fig, ax = plt.subplots(figsize=(6,6))
				self.samples.hist([i for i in variables], bins=bins, ax=ax)
		elif type(range) == tuple:
			if len(variables)>1:
				self.samples.hist([i for i in variables], range=range, bins=bins)
			else:
				fig, ax = plt.subplots(figsize=(6,6))
				self.samples.hist([i for i in variables], range=range, bins=bins, ax=ax)
		plt.ylabel("frequency")
		plt.xlabel("parameter value")
		plt.title(f"Frequency of sampled values of {[i for i in variables]}")
		plt.show()




	def vizkde(self, variable, showmax=False, range=(0.001,1)):
		"""Calculate and display a kernel density estimate for given variable
		
		Use SciPy to generate a gaussian KDE function, and calculate its value
		at each of 1000 bins in specified range. Plot these values.
		"""
		#TODO: truncate showmax digits. automatically plot max as line.
		#TODO: automatically detect good range for KDE.

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
			print(f"maximum density observed across 1000 bins was at x = {max_density:,.4f}")

	

	def summarize(self, suppress=False):
		"""Calculate summary statistics for all data.

		Set suppress = True to prevent display of calculated values.
		
		Determines mean, standard deviation, median, mode,
		highest posterior density, and 95% Crdible interval
		"""

		summarytable = pd.DataFrame(columns=['var','mean', 'std dev', 'median','mode','HPD','95CI'])
		for parameter in self.parameters:
			summarytable = summarytable.append(self.summarize_var(parameter), ignore_index=True)
		summarytable.set_index('var', inplace=True)
		del summarytable.index.name
		if not suppress:
			return summarytable


	def summarize_var(self, variable:str, ci="95"):
		"""Produce summary statistics for given variable.
		Determines mean, standard deviation, median, mode,
		highest posterior density, and 95% Crdible interval
		"""
		#TODO: save all of these calculated values in a summary dict.
		self.variable_statistics[variable] = {}
		details = self.variable_statistics[variable]
		var = self.get(variable)
		details['var'] = variable
		details['mean'] = np.mean(var)
		details['median'] = np.median(var)
		details['std dev'] = np.std(var)
		details['95CI'] = np.around(self.ci_percentile(var), decimals=3)
		details['HPD'] = np.around(self.ci_hpd(var), decimals=3)
		details['mode'] =  self.mode(var)
		return details
	
	def getstatistic(self, variable, statistic):
		"""Return indicated variable's indicated summary statistic"""
		return self.variable_statistics[variable][statistic]

	def ci_percentile(self,variable, bounds=(2.5,97.5)) -> (float, float):
		"""Return nearest observed data points to selected percentile boundaries"""
		return np.percentile(variable,bounds[0]),np.percentile(variable,bounds[1],interpolation='nearest')


	def ci_hpd(self, var_data, bins=1000, interval=95):
		"""Calculate the highest posterior density 95% credible interval for chosen variable
		
		Generates a 1000 bin histogram of data, sorts bins by
		how many data points are included, then collects bins
		one at a time until 95% of observations are in bins
		that have been collected.
		"""
		#TODO: Find a cleaner, more robust HPD function
		###### that describes multimodal data better.

		#build list of histogram bins sorted from most to least values captured.
		binvals, bins = np.histogram(var_data,bins)
		values_to_capture = (interval/100)*len(var_data)
		bintuples = [(bins[i],bins[i+1]) for i in range(len(bins)-1)]
		binpairs = zip(binvals, bintuples)
		sortedlist = sorted(binpairs, key=(lambda x : x[0]), reverse=True)

		#grow a list of bins until list contans >=95% of values
		values_captured = 0
		bins_included = []
		for histbin in sortedlist:
			values_captured += histbin[0]
			bins_included.append(histbin[1])
			if values_captured >= values_to_capture:
				break	

		#Record minimum of lowest bin and maximum of highest bin
		#Will produce bad interval for multimodal data
		span_captured = (min([i[0] for i in bins_included]), max([i[1] for i in bins_included]))	
		return span_captured

	def mode(self, variable, bins=100):
		"""Report most frequent sample of given variable.
		
		Because variable samples are typically continuous, mode is 
		approximated by reporting the densest bin in histogram of data.
		"""
		#TODO: Find out if mode makes any sense to report this way.

		bins = min([bins, len(variable)//5]) #enforce high bin count
		hist = np.histogram(variable,bins)
		maxindex = np.argmax(hist[0])
		maxbin = np.mean([hist[1][maxindex],hist[1][maxindex+1]])
		return maxbin
			

	def diagnostic(self):
		"""Calculate and report potential scale reduction factor"""
		rhats = {parameter : self.psrf(parameter) for parameter in self.parameters}
		maxhat = max(rhats, key = lambda val : rhats[val])
		if rhats[maxhat] < 1.05:
			print(f"all PSRF values < 1.05 | maximum PSRF: {maxhat} at {rhats[maxhat]}.")

	def showparameters(self):
		"""Print all tracked parameters in sample data"""
		print(','.join(i for i in self.parameters))

	def autocorr(self, variable, steps=10):
		"""Produce a plot showing the lagged autocorrelation of variable"""
		def lag(lag_steps, array):
			#lag k autocorrelation
			#as defined by https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm
			k = -lag_steps
			datamean = np.mean(array)
			calc_array = np.array([array,array])
			calc_array[1] = np.roll(calc_array[1],k)
			calc_array = calc_array[:,:k]
			a = calc_array[0,:]-datamean
			b = calc_array[1,:]-datamean
			return np.sum(a*b)/np.sum((array-datamean)**2)
		try:
			array = self.samples.loc[[0],[variable]].values
			lag_time = [lag(k, array) for k in range(1,steps+1)]
			lags = pd.DataFrame(lag_time, columns=["autocorrelation"])
			lags['lag'] = range(1,steps+1)
			sns.barplot(x='lag', y='autocorrelation', data=lags)
		except KeyError:
			print(f'Autocorrelation error: "{variable}" is not the name of any tracked parameter. \
			use showparameters() to check parameter names. ')

	def psrf(self, variable):
		"""Produce the Gelman-Rubin convergence diagnostic (Rhat)"""
		def gelrubin(myarray):
			# this function will not properly support chains of different lengths.
			#algorithm from http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf
			#and https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
			#chains = ndarray of size nsamples*nchains
			
			#TODO: make this calculation a little more elegant.
			chainmeans = np.mean(myarray, 0)
			totalmean = np.mean(myarray)
			nchains = np.shape(myarray)[1]#nchains
			nsamples = np.shape(myarray)[0]
			B = (nsamples/(nchains - 1))*sum([(chainmean - totalmean)**2 for chainmean in chainmeans])
			W = np.mean(np.var(myarray, 0))
			varhat = (1 - (1/nsamples))*W + (1/nsamples)*B
			rhat = np.sqrt(varhat/W)
			return rhat

		array = self.samples.loc[:,[variable]].unstack(level=0).values
		return gelrubin(array)




