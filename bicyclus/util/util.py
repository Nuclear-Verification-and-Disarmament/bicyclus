"""A collection of utility functions."""

import os

import arviz as az
import pymc as pm

from .log import log_print, task_identifier


def generate_start_values(sample_parameters, rng, n_chains=1):
    """Generate random start values.

    Parameters
    ----------
    sample_parameters : dict
        Must have the following form:
        {
            "distribution1_name": {
                "type": "Normal",  # Or any other valid PyMC distribution name
                "mu": 0,
                "sigma": 1
            },
            "distribution2_name": {
                "type": "Uniform",
                "lower": 0,
                "upper": 1
            }, ...
        }

    rng : int, RandomState or Generator
        Seed for the random number generator. We recommend to define a global
        numpy.Generator (e.g., using numpy.random.default_rng()) and pass this
        Generator to the function.

    n_chains : int, optional
        Number of sampled chains

    Returns
    -------
    dict or list of dicts
    """
    def genstart():
        start_values = {}
        for name, specs in sample_parameters.items():
            distribution = pm.__dict__[specs["type"]]
            start_values[name] = pm.draw(distribution.dist(
                **{k: v for k, v in specs.items() if k != "type"}),
                random_seed=rng)
        return start_values

    if n_chains == 1:
        return genstart()

    return [genstart() for i in range(0, n_chains)]


def sampling_parameter_to_pymc(name, value):
    """Parse a sampling parameter to a PyMC distribution.

    Parameters
    ----------
    name : str
        Name of the random variable.
    value : dict
        Must contain the 'type' key, which has the name of a PyMC distribution
        as value. The list of available distributions can be found on, e.g.,
        https://www.pymc.io/projects/docs/en/latest/api/distributions.html
        Note that PyMC capitalises their distribution names.
        All other kwargs are the distribution parameters. See example below.

    Returns
    -------
    pymc.Distribution

    Example
    -------
    .. code-block:: python
        # Create a standard Normal distribution.
        std_normal = {"type": "Normal", "mu": 0, "sigma": 1}
        dist = sampling_parameter_to_pymc("StandardNormalVariable", std_normal)
    """
    try:
        distribution = value["type"]
    except KeyError:
        msg = ("bicyclus.util.sampling_parameter_to_pymc: "
               "The 'value' dict must contain the 'type' key, which has the "
               "name of a PyMC distribution as value.")
        raise KeyError(msg)

    try:
        pymc_distribution = pm.distributions.__dict__[distribution](
            name=name, **{k: v for k, v in value.items() if k != "type"}
        )
    except TypeError:
        msg = ("bicyclus.util.sampling_parameter_to_pymc: "
               "Invalid parameter name in 'value' keyword.")
        raise TypeError(msg)

    return pymc_distribution


def samples_output_path(i, name="", dir_=None):
    """Generate the path and directories where the .cdf output is stored.

    Parameters
    ----------
    i : int
        Index of the run.

    name : str, optional
        Name of the reconstruction run.

    dir_ : str, optional
        Path where the .cdf file are stored. If None or '', then '$HOME/data'
        is used.

    Returns
    -------
    str
        Filename and path where output file is stored.
    """
    task_id = task_identifier()
    if dir_ is None or dir_ == "":
        dir_ = os.path.join(os.environ.get("HOME", None), "data")
    os.makedirs(dir_, exist_ok=True)

    return os.path.join(
        dir_, "cyclus_trace_{}_{}_{:04d}.cdf".format(name, task_id, i))


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

    i : int, optional
        Index
    """
    inference_data = pm.to_inference_data(trace)
    output_path = samples_output_path(i, args.run, dir_=args.output_path)

    log_print(f"Saving trace #{i} ({trace}) to file {output_path}")
    log_print(trace)
    log_print(az.summary(inference_data))
    location = inference_data.to_netcdf(output_path)
    log_print(f"Successfully saved trace #{i} to {location}")
