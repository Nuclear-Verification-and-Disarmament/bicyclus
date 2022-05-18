# Bicyclus minimal working example
This MWE will show the basics on how Bicyclus can be used.
We consider a very simple fuel cycle with a uranium source, an enrichment
facility, and two repositories for enriched and depleted uranium, respectively.

The goal in this scenario is to reconstruct the enrichment grade of the feed
uranium (the feed assay) through measurement of the total depleted uranium mass.
For simplicity, all other parameters are kept constant.

## File overview
The different files used are
- `cyclus_input.json`: Base Cyclus input file.
  This could also be an `.xml` or `.py` file.
- `run.py`: Driver file, where the specific behaviour is defined (what to
  extract from each simulation, how to calculate likelihoods, ...)
- `true_parameters.json`: The 'true' parameters, i.e., the ones used to generate
  the ground truth.
- `sampled_parameters.json`: The prior distributions of the sampled parameters.
