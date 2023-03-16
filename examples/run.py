#!/usr/bin/env python3

import json
import math
import os
from copy import deepcopy
from collections import namedtuple

import aesara.tensor as at
import arviz as az
import numpy as np
import pymc as pm

import bicyclus.blackbox
import bicyclus.util


RNG = np.random.default_rng(seed=12345)

# Using a namedtuple may be OP here but can improve clarity in more complex
# scenarios.
SimulationOutput = namedtuple("SimulationOutput", ("depleted_U_mass",))


class SampleCyclus(bicyclus.blackbox.CyclusCliModel):
    """This class is the main interface with Cyclus."""

    def __init__(self, true_parameters, sampled_parameters):
        self.true_parameters = true_parameters
        self.sampled_parameters = sampled_parameters
        self.current_parameters = None

        with open("cyclus_input.json", "r") as f:
            self.model = json.load(f)

        super().__init__()

    def mutate(self, sample=None):
        """Mutate the model, i.e. update its parameters.

        Parameters
        ----------
        sample : list
            Contains parameters corresponding to the parameters in
            `sampled_parameters`. The order must be identical to the
            alphabetically sorted order of keys in `sampled_parameters`.
        """
        # Set ground truth parameters.
        if sample is None:
            parameters = self.true_parameters

        else:
            # Alphabetical order. Only copy keys, as sampled_parameters
            # contain the sampling range.
            parameters = {k: None for k in self.sampled_parameters}
            # Unpack alphabetically sorted model parameters.
            for i, k in enumerate(sorted(self.sampled_parameters.keys())):
                parameters[k] = sample[i]
            bicyclus.util.log_print("Mutating: using parameters", parameters)

        # Change the isotopics of the feed uranium.
        self.mut_model = deepcopy(self.model)
        self.mut_model["simulation"]["recipe"][0]["nuclide"][0].update(
            comp=parameters["feed_assay"]
        )  # U235
        self.mut_model["simulation"]["recipe"][0]["nuclide"][1].update(
            comp=1.0 - parameters["feed_assay"]
        )  # U238
        self.current_parameters = parameters

    def result(self):
        """Extract the results from the last simulation."""
        target_facility_name = "DepletedUSink"
        rval = bicyclus.cyclus_db.multi_agent_concs(
            self.last_sqlite_file, [target_facility_name]
        )  # Returns concentrations and masses.
        total_mass = rval[1][target_facility_name]

        # Using a namedtuple may be OP here but can improve clarity in more
        # complex scenarios.
        sim_output = SimulationOutput(depleted_U_mass=total_mass)
        bicyclus.util.log_print("Simulation output:", sim_output)

        return sim_output


class IsotopeLikelihood(bicyclus.blackbox.LikelihoodFunction):
    """Class that calculates the likelihood."""

    def __init__(self, truth: SimulationOutput, rel_sigma=0.5):
        """Create a IsotopeLikelihood object.

        Parameters
        ----------
        truth : SimulationOutput
            The ground truth, i.e., simulation results using the true
            parameters.
        rel_sigma : float
            Relative sigma *in percent*.
        logdest : str (path) or None
            Path where the log will be stored.
        """
        self.truth = truth
        self.rel_sigma = rel_sigma

    def log_structured_sample(self, simout: SimulationOutput, likelihood: float):
        """Add results (concentrations, likelihood, parameters) to the log."""
        if self.structured_log is not None:
            d = {
                "likelihood": likelihood,
                "sink_masses": simout.additional_masses,
                "concentrations": {},
                "parameters": simout.parameters,
            }
            for sink in simout.composition.keys():
                for iso, concentration in simout.composition[sink].items():
                    if int(iso * 1e-4) in self.only_isos:
                        if sink not in d["concentrations"].keys():
                            d["concentrations"][sink] = {}
                        d["concentrations"][sink][iso] = concentration

            json.dump(d, self.structured_log)
            print("", file=self.structured_log)
            self.structured_log.flush()

    def log_likelihood(self, output: SimulationOutput):
        """Calculate the loglikelihood for a given measurement.

        The calculation considers the depleted U mass in this case, however,
        this can of course be generalised (taking into account, e.g., isotopic
        compositions).
        """
        # Convert relative sigma to absolute sigma.
        abs_sigma = lambda x: x * self.rel_sigma / 100.0

        # Normalise the differences. This is superfluous here but is important
        # in other scenarios to make the different likelihood contributions
        # comparable.
        centered = output.depleted_U_mass - self.truth.depleted_U_mass
        normalised = centered / abs_sigma(self.truth.depleted_U_mass)

        # Define the standard normal distribution. It is recommended *not* to
        # use PyMC's internal Normal distribution here, because it's evaluation
        # is much more expensive than the one shown below.
        std_normal = lambda x: math.exp(-(x**2) / 2) / (2 * math.pi) ** 0.5
        # TODO test if I have to insert a try-except clause here.
        try:
            llk = std_normal(normalised)
            logllk = math.log(llk)
        except ValueError as e:
            if llk < 1e-30:  # Arbitrarily chosen very small value.
                logllk = -np.inf
            else:
                raise e

        bicyclus.util.log_print("Mass loglikelihood for depleted U: " f"{logllk:.5e}")

        # Variable has to be returned as an array.
        return np.array(logllk)


def model(args):
    """Set up the model, priors, etc."""
    # Read prior distributions and groundtruths from files.
    with open(args.sample_parameters_file, "r") as f:
        sample_parameters = json.load(f)
    with open(args.true_parameters_file, "r") as f:
        true_parameters = json.load(f)

    # Set up the Cyclus blackbox and obtain the groundtruth.
    cyclus_model = SampleCyclus(true_parameters, sample_parameters)
    groundtruth = cyclus_model.run_groundtruth()
    bicyclus.util.log_print(f"Ground truth parameters are: {groundtruth}")

    # Set up the likelihood operator.
    loglikelihood_op = bicyclus.blackbox.CyclusLogLikelihood(
        IsotopeLikelihood(groundtruth, rel_sigma=args.rel_sigma),
        cyclus_model,
        memoize=True,
    )

    bicyclus.util.log_print("Building PyMC model.")
    bicyclus.util.log_print(
        "Sampling variables as follows:",
        [f"{k} => {v}" for (k, v) in sample_parameters.items()],
    )
    bicyclus.util.log_print(
        "The true parameters are:",
        [f"{k} => {v}" for (k, v) in true_parameters.items()],
    )

    with pm.Model() as pymc_model:
        # Transform the priors from the .json file to PyMC distributions.
        pymc_priors = {
            name: bicyclus.util.sampling_parameter_to_pymc(name, prior)
            for name, prior in sample_parameters.items()
        }

        bicyclus.util.log_print("Model variables:", pymc_priors)

        # Add the likelihood to the model.
        pm.Potential(
            "observed",
            loglikelihood_op(
                at.as_tensor_variable(
                    [pymc_priors[k] for k in sorted(pymc_priors.keys())]
                )
            ),
        )

        # Generate the initial values using the RNG to ensure reproducibility.
        initvals = bicyclus.util.generate_start_values(
            sample_parameters, RNG, args.chains
        )

    return pymc_model, initvals


def sample(args, pymc_model, initvals=None):
    """Sample the random variables and generate the trace(s)."""
    with pymc_model:
        # Algorithm must be one of the methods defined by PyMC, see
        # https://www.pymc.io/projects/docs/en/stable/api/samplers.html#step-methods
        if args.algorithm == "default":
            algorithm = pm.Slice()
        else:
            try:
                algorithm = pm.step_methods.__dict__[args.algorithm]()
            except KeyError:
                msg = (
                    "--algorithm must be one of the methods defined by "
                    "PyMC, see https://docs.pymc.io/en/v3/api/inference.html?highlight=step#step-methods. "
                    "Note that PyMC capitalises the first letter (e.g., "
                    "'Metropolis' instead of 'metropolis')."
                )
                raise KeyError(msg)

        # Use chunk sampling and the 'standard' PyMC sampling algorithm.
        if args.iter_sample <= 0:
            bicyclus.util.log_print("Starting sampling.")
            trace = None
            for i in range(0, args.iterations):
                bicyclus.util.log_print(
                    f"sampling iteration {i} at {args.samples} samples "
                    f"per iteration using {args.algorithm}, "
                    f"initial parameters {initvals}"
                )
                trace = pm.sample(
                    draws=args.samples,
                    tune=args.tune,
                    step=algorithm,
                    chains=args.chains,
                    cores=args.cores,
                    initvals=initvals,
                    compute_convergence_checks=False,
                    progressbar=False,
                    random_seed=RNG,
                    trace=trace,
                )
                bicyclus.util.save_trace(args, trace, i=i)

        # Use pm.iter_sample. We found that this algorithm is much slower for
        # unknown reasons.
        else:
            msg = (
                "Currently, using pm.iter_sample is not possible. See "
                "https://github.com/Nuclear-Verification-and-Disarmament/bicyclus/issues/14."
            )
            raise NotImplementedError(msg)

            bicyclus.util.log_print(
                f"Starting to sample iteratively (iter_sample: "
                f"{args.iter_sample}), initial parameters: {initvals}"
            )
            if args.chains > 1:
                bicyclus.util.log_print(
                    "WARNING: --chains > 1, but sampling iteratively. This "
                    "will not work -- sampling one chain only."
                )

            print(
                "\n\n\nWARNING: Reproducibility is not ensured in this "
                "pm.iter_sample example.\n\n\n"
            )
            sampler = pm.iter_sample(
                args.samples,
                algorithm,
                start=initvals,  # initvals[0] if type(initvals) is list else initvals,
                tune=args.tune,
            )
            sample_ix = 0
            saved_traces = 0
            # The sampling process starts here.
            for trace in sampler:
                sample_ix += 1
                bicyclus.util.log_print(
                    f"sampling: {saved_traces} {sample_ix}/{args.iter_sample}"
                )
                if sample_ix >= args.iter_sample:
                    save_trace(args, trace, i=saved_traces)
                    sample_ix = 0
                    saved_traces += 1
            bicyclus.util.save_trace(args, trace, i=saved_traces)

    bicyclus.util.log_print("Sampling finished!")


def main():
    """Main entry point of the script."""
    parser = bicyclus.util.ReconstructionParser()
    args = parser.get_args()

    bicyclus.util.write_to_log_file(
        run=args.run, outpath=args.log_path, debug=args.debug
    )

    pymc_model, initvals = model(args)
    sample(args, pymc_model, initvals)


if __name__ == "__main__":
    main()
