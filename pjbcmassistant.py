#this package contains convenience functions for handling
#the business of using Python 3.7 Jupyter Notebooks and
#the Python package pyjags, along with JAGS, to
#perform the code exercises in the textbook 
#Bayesian cognitive modeling: A practical course,
#by Lee, M. D., & Wagenmakers, E. J. (2014)

class model_spec:
"""
Handle loading, storing, providing, and displaying files that specify
a JAGS model.
"""
	def __init__(self, model:str, data:str, init:str) -> None:
		self.set(model, data, init)

	def load_model(self, filepath:str) -> None:
		with open(filepath, 'r') as modelfile:
			model_str modelfile.read()
		return model_str

	def load_data(self, filepath:str) -> None:
		data_str = ''
		with open(filepath, 'r') as datafile:
			data_str += [line for line in datafile]
		return model_str

	def load_init(self, filepath:str) -> None:
		data_str = ''
		with open(filepath, 'r') as datafile:
			data_str += [line for line in datafile]
		return model_str

	def load(self, model='':str, data='':str, init='':str) -> None:
		if model: self.model = self.load_model
		if data: self.data = self.load_data
		if init: self.init = self.load_init

	def set(self, attribute, value) -> None:
		attribute = value

	def get(self, attribute) -> str:
		return self.attribute

class viz:
"""
Handle common display functions for data in BCM exercises.
"""
	def __init__(self, dataframe) -> None:
		self.load(dataframe)

	def load(self, dataframe) -> None:
		self.df = dataframe

	def chains(self, dataframe, range=100):
		pass

	def hist(self, dataframe, bins=50):
		pass

	def kde(self, dataframe):
		pass















