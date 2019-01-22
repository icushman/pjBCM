#this package contains convenience functions for handling
#the business of using Python 3.7 Jupyter Notebooks and
#the Python package pyjags, along with JAGS, to
#perform the code exercises in the textbook 
#Bayesian cognitive modeling: A practical course,
#by Lee, M. D., & Wagenmakers, E. J. (2014)

import os
import pyjags
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

class SimpleSampler:
	def __init__(self, spec_string=None):
		if spec_string:
			self.model_spec = ModelHandler(spec_string)	

	def sample(self):
		self.pyjags_model = pyjags.Model(**self.model_spec.get_model_args())
		self.sample_data = self.pyjags_model.sample(**self.model_spec.get_sample_args())
		return SampleHandler(self.sample_data)

	def set_data(self, varname:str, value) -> None:
		self.model_spec.set_data(varname, value)

	def rebuild(self, spec_string):
		self.model_spec = ModelHandler(spec_string)


class ModelHandler:
	"""A class for defining the parameters used to generate a PyJAGS model.

	PyJAGS is a Python module designed to provide an interface to the JAGS
	Gibbs sampling application. This class contains methods for processing
	a simple string that specifies a model into a format PyJAGS will accept.

	For usage, print attributes `specification_template` and `model_template`.

	Attributes:
		specification_template: a demonstration of how to specify model properties
		model_template: a demonstration of how to define a JAGS model.
		model: a string representing the JAGS model to be passed to PyJAGS
	"""

	specification_template = """This is a template model specification string.
The model parser will ignore all text after a # symbol.

Copy and paste the following to specify a model and sampling rules,
and process it using my_model = model_handler(model_specification):
-----------------------------------------------
model_spec_string = \"\"\"
model: #model code or local .txt file
model{
  for (i in 1:observationCount){ 
    testScores[i] ~ dnorm(mu, lambda)
  }
  mu ~ dunif(0,10)
  lambda ~ dgamma(.001,.001)
}

settings:
chains = 3 #number of chains to run
samples = 1000 #number of samples per chain
thinning = 0 #number of samples to discard between recorded samples
burnin = 500 #number of burn-in samples per chain

data:
testScores = [1,2,3,4,5,6,7,8] # can be integer or list
observationCount = len(testScores) # variables can be assigned using Python expressions

sample: #model variables to record, one per line
mu
lambda
\"\"\"
-----------------------------------------------
"""

	def __init__(self, spec:str) -> None:
		"""Read in model specification string and store values to pass to PyJAGS"""
		self.configure(spec)
		self.modelfile= self.modelfile.rstrip()
		if self.modelfile.endswith('.txt'):
			self.model = self.load_model(self.modelfile)


		else:
			self.model = self.modelfile
# 			print("\nNo JAGS model was defined or selected. \
# Be sure to define a model before calling pyjags.Model")

	def load_model(self, filename:str) -> str:
		"""Load model file specified in spec string."""
		#TODO: clean up the logic around model specification.
		with open(os.path.join(os.getcwd(), filename), 'r') as modelfile:
			return modelfile.read()
	
	def set_model(self, model:str) -> None:
		"""Setter method for model"""
		self.model = model

	def configure(self, spec:str):
		"""Read spec string and bind all values needed for pyjags.Model
		
		Arguments:
			spec {str} -- a string representing the model specification
			(see ModelHandler.specification_template for expected format.)
		"""

		#TODO: Refactor this completely. It's nice to have the model spec look like
		#a simple settings file, but it's adding a lot of brittle complexity.
		#try to see who the users are, and whether this makes sense for them.
		#If so, rewrite it -- there are conventions for this kind of parsing,
		#and we're not really following them in this method.

		#initialize arguments to be passed to pyjags.Model
		self.modelfile = ''
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
			settingsline = settingsline.rstrip()
				

			#detect section headers	
			if settingsline.endswith(':'):
				section = settingsline

			elif len(settingsline) > 0:
				if section == 'model:': #'model' in section:
					self.modelfile += settingsline + '\n'
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
		"""Create keyword arguments dictionary for pyjags.model.Model
		
		Raises:
			self.ModelSpecificationError -- Some required model parameter was not provided.
		
		Returns:
			dict -- a dictionary mapping parameters expected by pyjags.Model
			to the arguments specified in the model handler. This dict can be
			passed directly to the model using **kwargs syntax.
		"""
		self.data = {item:np.array(self.data[item]).astype('float') for item in self.data}
		if self.model == None:
			raise self.Error("Please define a model in the model specification \
			 string or using model_handler.set_model()")

		masked_data = {data_var:np.ma.masked_invalid(self.data[data_var]) for data_var in self.data}
		model_args = {'code':self.model,
			'init':self.init,
			'data': masked_data if len(masked_data) > 0 else None,
			'chains':int(self.chains),
			'adapt':int(self.burnin),
		}
		return model_args

	def get_sample_args(self) -> dict:
		"""Create keyword arguments dictionary for pyjags.model.Model.sample
		
		Returns:
			dict -- a dictionary mapping parameters expected by pyjags. Sample
			to the arguments specified in the model handler. This dict can be
			passed directly to the model using **kwargs syntax.
		"""
		sample_args = {'vars':self.sample,
			'iterations':int(self.samples),
		}
		if int(self.thinning) > 0:
			sample_args['thin'] = int(self.thinning)
		return sample_args

	def set_data(self, varname:str, value) -> None:
		"""Set model's data from outside the specification string.
		
		Arguments:
			varname {str} -- the model variable name you wish to assign
			value {int or list} -- the value to you wish a model variable to take
		"""
		self.data[varname] = value

	def init_chain(self, varname:str, init_values:list) -> None:
		"""Set the initial values of selected chain using values or a distribution.
		
		Arguments:
			varname {str} -- The name of a model variable whose initial value you wish\
			to fix.
			init_values {list, int, or string} -- If init_values is a list, each item\
in the list will be assigned to a chain. If single value, each chain will be initialized\
with that value. If string, value must indicate a distribution from numpy.random that produces\
a dingle number, e.g., `'uniform(0,5)'` or `'gamma(1,1)'`.
		
		Raises:
			self.Error -- Unacceptable values of `init_values` argument raise exceptions.
		"""
		#Note: It would be cool to know all of the model variables at this stage
		#but that would require some obscene parsing, a rework of the logic flow,
		#or a 'trial run' model initialization...

		self.chains = int(self.chains)

		if type(init_values) not in (np.ndarray, list, str, float, int):
			raise self.Error(f"ModelHandler.init_chain: expected initial values as list, \
int, float, or string representing numpy distribution. See help for usage.")

		if (type(init_values) in (np.ndarray, list) and
				len(init_values) != self.chains):
					raise self.Error(f"ModelHandler.init_chain: expected {self.chains} values\
(one per chain), but only received {len(init_values)}. If you wish to have all chains\
use same initial value, pass it as int, float, or distribution, rather than as a list.")


		if type(init_values) == str:
			source_distribution = lambda : eval('np.random.'+init_values)
			try:
				if type(source_distribution()) not in (float, int):
					raise self.Error(f"ModelHandler.init_chain: string input for init_values \
did not specify appropriate numpy.random distribution.")
			except:
				raise self.Error(f"ModelHandler.init_chain: Invalid string input for init_values. \
Argument must be string specifying an appropriate numpy.random distribution, e.g., 'uniform(0,1)'.")
			init_values = [source_distribution() for i in range(self.chains)]

		elif type(init_values) in (int, float):
			init_values = [init_values for i in range(self.chains)]

		if not self.init:
			self.init = [{varname:init_value} for init_value in init_values]

		else:
			new_init_list = [{varname:init_value} for init_value in init_values]
			self.init = [{**new_init_list[i], **self.init[i]} for i in range(len(self.init))]


	class Error(Exception):
		"""Raised when model specification parameters are incorrect."""
		pass

	

class SampleHandler:
	"""A class for handling samples produced by the PyJAGS interface.

	This class includes numerous methods for visualizing data and
	evaluating sample convergence.
	"""
	
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

				#what follows handles nested variables >= 3 dimensions
				#this is very hacky -- should ideally be handled by the same
				#code as all other cases.
				newkeys = {}
				while any(len(np.shape(data[key])) > 2 for key in data):
					killkeys = set()
					for key in data:
						if len(np.shape(data[key])) > 2:
							killkeys.add(key)
							for i in range(np.shape(data[key])[0]):
								newkeys[f"{key}_{str(i)}"] = data[key][i,:,:]
					data = {**data, **newkeys}
					for key in killkeys:
						del(data[key])

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
		"""Return all sample values of given variable
		
		Arguments:
			variable {str} -- A string referring to a model variable.
		
		Returns:
			numpy.array -- an array of samples from the specified variable.
		"""
		return self.samples[variable].values
		
		
	def head(self):
		"""Print first 5 rows of samples dataframe"""
		print(self.samples.head())


	def vizchains(self, *variables:str, range=100):
		"""Generate line plot for the first [range] samples of given variable(s)
		
		Arguments:
			variables {[str]} -- The model variables whose chains you wish to visualize. 

		Keyword Arguments:
			range {int} -- The number of samples to visualize (default: {100})
		"""
		if len(variables) == 0:
			raise SampleHandler.Error("SampleHandler.vizchains: requires one or more sample variable names as arguments.")
		if not all(variable in self.parameters for variable in variables):
			raise SampleHandler.Error("SampleHandler.vizchains: every argument must be the name of a sampled variable.")

		fig, ax = plt.subplots(figsize=(8,3))
		self.samples.groupby('chain')[[i for i in variables]].apply(lambda x: x).unstack(level=0)[:range].plot(ax=ax)
		plt.ylabel("parameter value")
		plt.title(f"First {range} samples of {[i for i in(variables)]}")
		plt.show()

	def vizjoint(self, variable1, variable2):
		"""Produce joint plot of given variables.
		
		Arguments:
			variable1 {str} -- the first model variable to plot
			variable2 {str} -- the second model variable to plot
		"""
		joint = sns.jointplot(variable1, variable2, data=self.samples, color="grey");

	def vizhist(self, *variables, bins=50, range:(int,int)=None):
		"""Produce histogram of given variable(s)
		
		Keyword Arguments:
			bins {int} -- number of histogram bins to display (default: {50})
			range {(int,int)} -- the lower and upper bounds for the
			displayed x axis (default: {None})
		"""
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
		
		Arguments:
			variable {str} -- the model variable to plot
		
		Keyword Arguments:
			showmax {bool} -- print the value of the KDE maximum (default: {False})
			range {(float,float)} -- the lower and upper bounds of resulting
			x axis (default: {(0.001,1)})
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
		
		Keyword Arguments:
			suppress {bool} -- prevent calculated values from printing
			to console (default: {False})
		
		Returns:
			pandas dataframe -- a dataframe listing the summary statistics for all
			tracked variables.
		"""
		#TODO: I don't think the suppress option really makes sense.
		# Revist where this is used.
		summarytable = pd.DataFrame(columns=['var','mean', 'std dev', 'median','mode','HPD','95CI'])
		for parameter in self.parameters:
			summarytable = summarytable.append(self.summarize_var(parameter), ignore_index=True)
		summarytable.set_index('var', inplace=True)
		del summarytable.index.name
		if not suppress:
			return summarytable


	def summarize_var(self, variable:str):
		"""Produce summary statistics for given variable.
		Determines mean, standard deviation, median, mode,
		highest posterior density, and 95% Credible interval.
		
		Arguments:
			variable {str} -- the model variable name to be summarized
		
		Returns:
			dict -- A map of summary statistics to their calculated values.
		"""
		#TODO: save all of these calculated values in a summary dict.
		self.variable_statistics[variable] = details = {}
		#details = self.variable_statistics[variable]
		var = self.get(variable)
		details['var'] = variable
		details['mean'] = np.mean(var)
		details['median'] = np.median(var)
		details['std dev'] = np.std(var)
		details['95CI'] = np.around(self._ci_percentile(var), decimals=3)
		details['HPD'] = np.around(self._ci_hpd(var), decimals=3)
		details['mode'] =  self._mode(var)
		return details
	
	def get_statistic(self, variable, statistic):
		"""Return indicated variable's indicated summary statistic

		This will only work if SampleHandler.summarize_var has already been called on
		the selected variable either manually or via SampleHandler.summarize.
		
		Arguments:
			variable {str} -- model variable name
			statistic {str} -- one of 'var', 'mean', 'median', 'std dev',
			'95CI', 'HPD', 'mode'.
		
		Returns:
			float or (float,float) -- the requested statistic for indicated variable.
		"""
		return self.variable_statistics[variable][statistic]

	def _ci_percentile(self, variable, bounds=(2.5,97.5)) -> (float, float):
		"""Return nearest observed data points to selected percentile boundaries
		
		Arguments:
			variable {np array} -- the array or data from a variable of interest.
			bounds {(float, float)} -- the lower and upper confidence interval (0 to 100)
		
		Returns:
			(float, float) -- the lower and upper percentile for the selected data's
			samples.
		"""
		return np.percentile(variable,bounds[0]),np.percentile(variable,bounds[1],interpolation='nearest')


	def _ci_hpd(self, var_data, bins=1000, interval=.95):
		"""Calculate the highest posterior density credible interval for chosen variable.
		
		Generates a histogram of data, sorts the bins by
		how many data points are included, then collects bins
		one at a time until 95% of observations are in bins
		that have been collected.

		As implemented, this will produce a bad interval for
		multimodal data, inclusive of intermediate regions with
		low density.

		Algorithm appears standard, but this version was taken from
		https://stats.stackexchange.com/questions/252988/highest-density-interval-in-stan
		written by user `LukasNeugebauer`

		Arguments:
			var_data {np array} -- the array or data from a variable of interest.
		
		Keyword Arguments:
			bins {int} -- The number of bins to use in HPD calculation (default: {1000})
			interval {int} -- The desired percent credible interval (default: {95})
		
		Returns:
			(float, float) -- The lower and upper bounds of the selected variable's HPD.
		"""
		#TODO: Find a more robust HPD function
		###### that describes multimodal data better.
		
		#algorithm taken from Stack Exchange user LukasNeugebauer
		#--------------------------------------------------------
		chain = var_data
		# def computeHDI(chain, interval = .95):
		# sort chain using the first axis which is the chain
		chain = np.array(sorted(chain))
		# how many samples did you generate?
		nSample = chain.size    
		# how many samples must go in the HDI?
		nSampleCred = int(np.ceil(nSample * interval))
		# number of intervals to be compared
		nCI = nSample - nSampleCred
		# width of every proposed interval
		width = np.array([chain[i+nSampleCred] - chain[i] for  i in range(nCI)])
		# index of lower bound of shortest interval (which is the HDI) 
		best  = width.argmin()
		# put it in a dictionary
		HDI   = {'Lower': chain[best], 'Upper': chain[best + nSampleCred], 'Width': width.min()}
		bounds = (HDI['Lower'],HDI['Upper'])
		return bounds



	def _mode(self, variable, bins=100):
		"""Report most frequent sample of given variable.
		
		Because variable samples are typically continuous, mode is 
		approximated by reporting the densest bin in histogram of data.
		This measure is sensitive to bin count.
		
		Arguments:
			variable {np array} -- the array or data from a variable of interest.
		
		Keyword Arguments:
			bins {int} -- Number of bins to use when determining mode. (default: {100})
		
		Returns:
			float -- The average of boundary values of the bin containing the most samples.
		"""
		#TODO: Find out if mode makes any sense to report this way.
		bins = min([bins, len(variable)//5]) #enforce reasonable bin count
		hist = np.histogram(variable,bins)
		maxindex = np.argmax(hist[0])
		maxbin = np.mean([hist[1][maxindex],hist[1][maxindex+1]])
		return maxbin
			
	def show_parameters(self):
		"""Print all tracked parameters in sample data"""
		print(','.join(i for i in self.parameters))

	def autocorr(self, *variables, steps=10):
		"""Produce a plot showing the lag k autocorrelation of selected variable

		The alogrithm for autocorrelation is produced according to the description
		reported at https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm
		
		Arguments:
			variable {str} -- The name of a tracked model variable
		
		Keyword Arguments:
			steps {int} -- The amount of lagged time steps for which to
			report autocorrelation
			(default: {10})
		"""
		if len(variables) == 0:
			raise SampleHandler.Error("SampleHandler.autocorr: requires one \
or more sample variable names as arguments.")
		
		def _lag(lag_steps, array):
			"""Helper function to implement autocorrelation algorithm."""
			k = -lag_steps
			datamean = np.mean(array)
			calc_array = np.array([array,array])
			calc_array[1] = np.roll(calc_array[1],k)
			calc_array = calc_array[:,:k]
			a = calc_array[0,:]-datamean
			b = calc_array[1,:]-datamean
			return np.sum(a*b)/np.sum((array-datamean)**2)
		for variable in variables:
			try:
				array = self.samples.loc[[0],[variable]].values
				lag_time = [_lag(k, array) for k in range(1,steps+1)]
				lags = pd.DataFrame(lag_time, columns=["autocorrelation"])
				lags['lag'] = range(1,steps+1)
				sns.barplot(x='lag', y='autocorrelation', data=lags)
				plt.title(f"Autocorrelation: First {steps} steps of {variable}")
				plt.show();
			except KeyError:
				print(f'Autocorrelation error: "{variable}" is not the name of any tracked \
parameter. use show_parameters() to check parameter names. ')

	def diagnostic(self):
		"""Calculate all Rhat values, report the highest.

		See SampleHandler.psrf for implementation details.
		"""
		
		nans_present = False
		rhats = {parameter : self.psrf(parameter) for parameter in self.parameters}

		#filter out NaN values for the purpose of finding max rhat (there's a prettier way)
		if any(np.isnan(value) for value in rhats.values()):
			nans_present = True
			rhats = {key:rhats[key] for key in rhats.keys() if not np.isnan(rhats[key])}

		maxhat = max(rhats, key = lambda val : rhats[val])

		if rhats[maxhat] < 1.05 and not nans_present:
			print(f"all PSRF values < 1.05 | maximum PSRF: {maxhat} at {rhats[maxhat]}.")
		elif rhats[maxhat] < 1.05 and nans_present:
			print(f"all PSRF values < 1.05 (but some nodes returned NaN) | maximum PSRF: {maxhat} at {rhats[maxhat]}.")
		else:
			print(f"Evidence of poor mixing.  Not all PSRF values under 1.05 \n maximum PSRF: {maxhat} at {rhats[maxhat]}.")

	def psrf(self, variable):
		"""Produce the Gelman-Rubin convergence diagnostic (Rhat) for selected variable.

		This statistic, referred to as the Gelman-Rubin convegence diagnostic,
		potential scale reduction factor, or R-hat, compares variance within
		sample chains to variance across sample chains.
		The algorithm used was drawn from:
		http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf
		https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
		Note: this algorithm will not properly support chains of different lengths.
		
		Arguments:
			variable {str} -- Name of tracked model variable.
		
		Returns:
			int -- The Rhat value for the selected variable.
		"""
		def _gelrubin(sample_array):
			"""Implements the Gelman-Rubin potential scale reduction factor algorithm"""
			#TODO: make this calculation a little more elegant.
			chainmeans = np.mean(sample_array, 0)
			totalmean = np.mean(sample_array)
			nchains = np.shape(sample_array)[1]
			nsamples = np.shape(sample_array)[0]
			B = (nsamples/(nchains - 1))*sum([(chainmean - totalmean)**2 for chainmean in chainmeans])
			W = np.mean(np.var(sample_array, 0))
			varhat = (1 - (1/nsamples))*W + (1/nsamples)*B
			rhat = np.sqrt(varhat/W)
			return rhat

		array = self.samples.loc[:,[variable]].unstack(level=0).values
		return _gelrubin(array)

	class Error(Exception):
		pass



