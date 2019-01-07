# ModelHandler
```python
ModelHandler(self, spec: str) -> None
```
A class for defining the parameters used to generate a PyJAGS model.

PyJAGS is a Python module designed to provide an interface to the JAGS
Gibbs sampling application. This class contains methods for processing
a simple string that specifies a model into a format PyJAGS will accept.

For usage, print attribute `specification_template`.

Attributes:
 specification_template: a demonstration of how to specify model properties 
 model: a string representing the JAGS model to be passed to PyJAGS

## load_model
```python
ModelHandler.load_model(self, filename: str) -> str
```
Load model file specified in spec string.
## set_model
```python
ModelHandler.set_model(self, model: str) -> None
```
Setter method for model
## configure
```python
ModelHandler.configure(self, spec: str)
```
Read spec string and bind all values needed for pyjags.Model

Arguments:
 spec {str} -- a string representing the model specification
 (see ModelHandler.specification_template for expected format.)

## get_model_args
```python
ModelHandler.get_model_args(self)
```
Create keyword arguments dictionary for pyjags.model.Model

Raises:
 self.ModelSpecificationError -- Some required model parameter was not provided.

Returns:
 dict -- a dictionary mapping parameters expected by pyjags.Model
 to the arguments specified in the model handler. This dict can be
 passed directly to the model using **kwargs syntax.

## get_sample_args
```python
ModelHandler.get_sample_args(self) -> dict
```
Create keyword arguments dictionary for pyjags.model.Model.sample

Returns:
 dict -- a dictionary mapping parameters expected by pyjags. Sample
 to the arguments specified in the model handler. This dict can be
 passed directly to the model using **kwargs syntax.

## set_data
```python
ModelHandler.set_data(self, varname: str, value) -> None
```
Set model's data from outside the specification string.

Arguments:
 varname {str} -- the model variable name you wish to assign
 value {int or list} -- the value to you wish a model variable to take

## ModelSpecificationError
```python
ModelHandler.ModelSpecificationError(self, /, *args, **kwargs)
```
Raised when model specification parameters are incorrect.
# SampleHandler
```python
SampleHandler(self, samples) -> None
```
A class for handling samples produced by the PyJAGS interface.

This class includes numerous methods for visualizing data and
evaluating sample convergence.

## get
```python
SampleHandler.get(self, variable)
```
Return all sample values of given variable

Arguments:
 variable {str} -- A string referring to a model variable.

Returns:
 numpy.array -- an array of samples from the specified variable.

## head
```python
SampleHandler.head(self)
```
Print first 5 rows of samples dataframe
## vizchains
```python
SampleHandler.vizchains(self, *variables, range=100)
```
Generate line plot for the first [range] samples of given variable(s)

Arguments:
 variables {[str]} -- The model variables whose chains you wish to visualize.

Keyword Arguments:
 range {int} -- The number of samples to visualize (default: {100})

## vizjoint
```python
SampleHandler.vizjoint(self, variable1, variable2)
```
Produce joint plot of given variables.

Arguments:
 variable1 {str} -- the first model variable to plot
 variable2 {str} -- the second model variable to plot

## vizhist
```python
SampleHandler.vizhist(self, *variables, bins=50, range: (<class 'int'>, <class 'int'>) = None)
```
Produce histogram of given variable(s)

Keyword Arguments:
 bins {int} -- number of histogram bins to display (default: {50})
 range {(int,int)} -- the lower and upper bounds for the
 displayed x axis (default: {None})

## vizkde
```python
SampleHandler.vizkde(self, variable, showmax=False, range=(0.001, 1))
```
Calculate and display a kernel density estimate for given variable

Use SciPy to generate a gaussian KDE function, and calculate its value
at each of 1000 bins in specified range. Plot these values.

Arguments:
 variable {str} -- the model variable to plot

Keyword Arguments:
 showmax {bool} -- print the value of the KDE maximum (default: {False})
 range {(float,float)} -- the lower and upper bounds of resulting
 x axis (default: {(0.001,1)})

## summarize
```python
SampleHandler.summarize(self, suppress=False)
```
Calculate summary statistics for all data.

Set suppress = True to prevent display of calculated values.

Determines mean, standard deviation, median, mode,
highest posterior density, and 95% Crdible interval

Keyword Arguments:
 suppress {bool} -- prevent calculated values from printing
 to console (default: {False})

Returns:
 pandas dataframe -- a dataframe listing the summary statistics for all
 tracked variables.

## summarize_var
```python
SampleHandler.summarize_var(self, variable: str)
```
Produce summary statistics for given variable.
Determines mean, standard deviation, median, mode,
highest posterior density, and 95% Credible interval.

Arguments:
 variable {str} -- the model variable name to be summarized

Returns:
 dict -- A map of summary statistics to their calculated values.

## get_statistic
```python
SampleHandler.getstatistic(self, variable, statistic)
```
Return indicated variable's indicated summary statistic

This will only work if SampleHandler.summarize_var has already been called on
the selected variable either manually or via SampleHandler.summarize.

Arguments:
 variable {str} -- model variable name
 statistic {str} -- one of 'var', 'mean', 'median', 'std dev',
 '95CI', 'HPD', 'mode'.

Returns:
 float or (float,float) -- the requested statistic for indicated variable.

## showparameters
```python
SampleHandler.showparameters(self)
```
Print all tracked parameters in sample data
## autocorr
```python
SampleHandler.autocorr(self, variable, steps=10)
```
Produce a plot showing the lag k autocorrelation of selected variable

The alogrithm for autocorrelation is produced according to the description
reported at https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm

Arguments:
 variable {str} -- The name of a tracked model variable

Keyword Arguments:
 steps {int} -- The amount of lagged time steps for which to
 report autocorrelation
 (default: {10})

## diagnostic
```python
SampleHandler.diagnostic(self)
```
Calculate all Rhat values, report the highest.

See SampleHandler.psrf for implementation details.

## psrf
```python
SampleHandler.psrf(self, variable)
```
Produce the Gelman-Rubin convergence diagnostic (Rhat) for selected variable.

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

