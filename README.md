# Bayesian Cognitive Modeling with `pyjgags`

This is a project to port the code examples in Lee and Wagenmakers' 2013 textbook [Bayesian Cognitive Modeling: A Practical Course](https://bayesmodels.com/) from MATLAB into Python, using the [pyjags](https://github.com/tmiasko/pyjags) package for interfacing Python code with the `JAGS` Gibbs sampling application.

The main contribution is the module `pjbcmassistant`, which contains convenience classes `ModelHandler` and `SampleHandler` for easily interfacing with `pyjags`, and for performing basic analysis on the model samples it produces.

### Quick Start:
The notebook [PyJAGS-BCM Usage Guide](https://github.com/icushman/pjBCM/blob/master/GettingStarted/PyJAGS-BCM%20Usage%20Guide.ipynb) provides a demonstration and overview of how to use the module for building and analyzing models.

See the [full documentation](https://github.com/icushman/pjBCM/blob/master/docs.md) for details on all methods provided by the module.

*NOTE:*  
In addition to dependencies listed in requirements.txt, this module requires that `JAGS` (which is not a Python package) be successfully installed on your system prior to configuring `pyjags`. See the [JAGS website](http://mcmc-jags.sourceforge.net/) and the [pyjags installation instructions](https://pyjags.readthedocs.io/en/latest/getting_started.html) for details.
