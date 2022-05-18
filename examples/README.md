# Bicyclus minimal working example
This MWE will show the basics on how Bicyclus can be used.
The different files used are
- `cyclus_input.json`: Cyclus input file, which could also be an `.xml` or `.py`
  file)
- `run.py`: Driver file, where the specific behaviour is defined (what to
  extract from each simulation, how to calculate likelihoods, ...)
- `true_parameters.json`: The 'true' parameters, i.e., the ones used to generate
  the ground truth.
- `sampled_parameters.json`: The prior distributions of the sampled parameters.
