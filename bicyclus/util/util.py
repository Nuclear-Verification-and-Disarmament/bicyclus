"""A collection of utility functions."""

import arviz as az
import pymc3 as pm

from log import log_print


def sampling_parameter_to_pymc(name, value):
    """Parse a sampling parameter to a PyMC3 distribution.

    Parameters
    ----------
    name : str
        Name of the random variable.
    value : dict
        Must contain the 'type' key, which has the name of a PyMC3 distribution
        as value. The list of available distributions can be found on, e.g.,
        https://docs.pymc.io/en/v3/api/distributions.html. Note that PyMC
        capitalises their distribution names.
        All other kwargs are the distribution parameters. See example below.

    Returns
    -------
    pymc3.Distribution

    Example
    -------
    .. code-block:: python
        # Create a standard Normal distribution.
        std_normal = {"type": "Normal", "mu": 0, "sigma": 1}
        dist = sampling_parameter_to_pymc("StandardNormalVariable", std_normal)
    """
    try:
        distribution = value["type"]
        del value["type"]
    except KeyError:
        msg = ("bicyclus.util.sampling_parameter_to_pymc: "
               "The 'value' dict must contain the 'type' key, which has the "
               "name of a PyMC3 distribution as value.")
        raise KeyError(msg)

    try:
        pymc_distribution = pm.distributions.__dict__[distribution](name,
                                                                    **value)
    except TypeError:
        msg = ("bicyclus.util.sampling_parameter_to_pymc: "
               "Invalid parameter name in 'value' keyword.")
        raise TypeError(msg)

    return pymc_distribution

def save_trace(args, trace, i=0):
    """Save a trace as .cdf file.

    Parameters
    ----------
    args : parsed arguments
        The (parsed) arguments as defined in SamplingParser, see
        /bicyclus/util/parsers.py. This functioning may be adapted later to
        become independent of the Bicyclus-argparser.

    trace : pymc3 MultiTrace object
        The trace to be saved.
    i : int
        Index
    """
    output_path = samples_output_path(i, args.run, default=args.output_path)
    log_print(f"Saving trace #{i} ({trace}) to file {output_path}")
    log_print(trace)
    log_print(az.summary(az.from_pymc3(trace)))
    location = az.from_pymc3(trace,
                             density_dist_obs=False).to_netcdf(output_path)
    log_print(f"Successfully saved trace #{i} to {location}")
