# Bicyclus
Bicyclus is a Bayesian Inference module for Cyclus.
Specifically, it is an interface written in Python3 that connects
[Cyclus](https://fuelcycle.org/), a nuclear fuel simulator, with
[PyMC](https://docs.pymc.io/en/v3/) (formerly PyMC3), a Python package for
Bayesian statistics using Markov chain Monte Carlo algortihms.

> Bicyclus, Bicyclus, Bicyclus  
> I want to use my Bicyclus  
> I want to use Cyclus  
> I want to infer with Cyclus  
> I want to infer what I like.  
>
> _Freddie Mercury (somewhen, somewhere, probably)_

## Features
- Run inference processes on nuclear fuel cycles simulated with Cyclus.
- Store quantities of interest and simulation output that are not part of the
  inference process, in separate files.
  These can be used for later analysis.
- Use PyMC distributions as priors (WIP, currently restricted to a subset of
  distributions).
- Features utility functions for post-processing and plotting posterior
  distributions.

## How-to
### Requirements
Bicyclus has only been tested using cherry-picked versions of `PyMC3` and
`Arviz`, with certain patches applied.
Using standard PyMC and Arviz versions may lead to errors or unexpected
behaviour and is therefore *not* recommended.
The following table lists the software requirements.

__Update__: This version is currently being tested using Python 3.10.6, PyMC
(as opposed to PyMC3) version 4.2.0. and Arviz 0.12.1.
PyMC and Arviz have been installed using conda and the conda-forge channel.

| Name | Version | Notes |
|:-----|---:|:---|
| [PyMC](https://www.pymc.io/welcome.html) | `4.2.0` | Currently being tested |
| [Aesara](https://aesara.readthedocs.io/en/latest/) | `2.8.2` | Currently being tested, included in PyMC |
| [Arviz](https://python.arviz.org/en/latest/index.html) | `0.12.1` | Currently being tested |
| [NumPy](https://numpy.org/doc/stable/index.html) | n/a | no specific version |
| [SciPy](https://docs.scipy.org/doc/scipy/index.html)| n/a | no specific version |
| [Pandas](https://pandas.pydata.org/)| n/a | no specific version |
| [matplotlib](https://matplotlib.org/)| n/a | no specific version |
| [seaborn](https://seaborn.pydata.org/) | | |

### Installation
__Please note__: Assuming the tests with the latest PyMC version work, then
step can be skipped.
1. Install the cherry-picked versions of Arviz and PyMC3 (listed above):
   ```bash
   $ git clone https://git.rwth-aachen.de/lewin/pymc3.git
   $ cd pymc3
   $ git checkout lewin
   $ pip3 install .
   $ git clone https://git.rwth-aachen.de/lewin/arviz.git
   $ cd arviz
   $ git checkout lewin
   $ pip3 install .
   ```
2. Install Bicyclus:
   ```bash
   $ git clone https://github.com/Nuclear-Verification-and-Disarmament/bicyclus.git
   $ cd bicyclus
   $ pip3 install .
   ```

### Tutorial
A minimum working example (MWE) can be found in the [`examples`](/examples)
directory.
At the moment, Bicyclus can be used through a driver script that has to be
written on a case-by-case basis.
However, the MWE provided should be a good starting point for any new driver script.

## Legacy code
Originally, this work was developed as part of a Bachelor's thesis by Lewin
Bormann.
If you are interested in the git blame and log beyond this repository's initial
commit, please visit the
[original repository](https://git.rwth-aachen.de/nvd/fuel-cycle/bayesian-cycle/).
That repository also contains two applications of complex nuclear fuel cycles
and reconstruction scenarios.

## Pitfalls
- Depending on the runtime of one Cyclus run, `subprocess`'s timeout value has
  to be adapted.
  It is defined in `bicyclus/blackbox/blackbox.py`
  (`CyclusCliModel.simulate`) and is currently set to 300 seconds (as of 2022/04/14).
