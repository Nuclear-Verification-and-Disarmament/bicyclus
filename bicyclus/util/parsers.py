""""Predefined parsers to start Bicyclus runs or to analyse its output."""

import argparse
import os
from abc import ABC, abstractmethod

class BaseParser(ABC):
    def __init__(self, **kwargs):
        """Initialise the parser with optional arguments."""
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)
        self.add_args()

    @abstractmethod
    def add_args(self):
        """Implement the different arguments of the parser here."""
        pass

    def get_args(self, parsed=True):
        """Return the parser.

        Parameters
        ----------
        parsed : bool, optional
            If true (default), the arguments are parsed and returned. If false,
            the parser is returned. This can be useful if the user wants to add
            additional arguments before parsing.

        Returns
        -------
        See description above.
        """
        if parsed:
            return self.parser.parse_args()
        return self.parser

class SamplingParser(BaseParser):
    """"Parser used in a driver file to start Bicyclus simulations."""
    def __init__(self, description="Run Bicyclus simulations.", **kwargs):
        super().__init__(**kwargs)

    def add_args(self):
        user = os.environ["USER"]
        default_dir = os.getcwd()
        default_data_dir = os.path.join(default_dir, "data")
        default_log_dir = os.path.join(default_dir, "job_output")

        if not os.path.isdir(default_data_dir):
            os.mkdir(default_data_dir)
        if not os.path.isdir(default_log_dir):
            os.mkdir(default_log_dir)

        # Job parameters
        job_group = self.parser.add_argument_group("Job parameters")
        job_group.add_argument(
            "--cores", type=int, default=1,
            help="Number of cores to be used, corresponding to the number of "
                 "chains to be run in parallel. Ideally, it should be equal "
                 "to '--chains'.")
        job_group.add_argument(
            "--debug", default=False, action="store_true",
            help="If set, print all output to STDOUT instead of storing"
                 " it in the log file. Overrides '--log-path' (if set).")
        job_group.add_argument(
            "--index", type=int, default=0, help="Instance index")
        job_group.add_argument(
            "--log-path", type=str, default=default_log_dir,
            help="Output path for the job logs.")
        job_group.add_argument(
            "--output-path", type=str, default=default_data_dir,
            help="Output path for the sampled .cdf files.")
        job_group.add_argument(
            "--run", type=str, help="Name of this run.")

        # Sampling parameters
        sampling_group = self.parser.add_argument_group("Sampling parameters")
        sampling_group.add_argument(
            "--algorithm", default="default",
            help="PyMC3 sampling algorithm to be used, e.g., 'Slice' or "
                 "'Metropolis'. See https://docs.pymc.io/en/v3/api/inference.html?highlight=step#step-methods "
                 "for a list of available samplers.")
        sampling_group.add_argument(
            "--chains", type=int, default=1, help="Number of chains")
        sampling_group.add_argument(
            "--iterations", type=int, default=5,
            help="Number of successive iterations that should be run, each "
                 "--samples. Total number of samples is "
                 "`--samples * --iterations`.")
        sampling_group.add_argument(
            "--iter-sample", type=int, default=0,
            help="Experimental: If > 0, use pm.iter_sample() and save a "
                 "trace every --iter-sample iterations. WARNING: We found "
                 "pm.iter_sample to be much slower than the default "
                 "pm.sample.")
        sampling_group.add_argument(
            "--samples", type=int, default=400,
            help="Number of samples per chain per iteration")
        sampling_group.add_argument(
            "--tuning-samples", type=int, default=-1,
            help="Number of tuning samples. If < 0 (default), then it is set "
                 "to 'number of samples' / 10.")  # This has to be done in the driver script!

        # Additional parameters
        additional_group = self.parser.add_argument_group(
            "Additional parameters")
        additional_group.add_argument(
            "--rel-sigma", type=float, default=0.5,
            help="Relative sigma for calculation of likelihoods, in percent.")
        additional_group.add_argument(
            "--sample-parameters-file",  type=str,
            default=os.path.join(default_dir, "sample_parameters.json"),
            help="JSON file containing the sampled prior distributions.")
        additional_group.add_argument(
            "--true-parameters-file", type=str,
            default=os.path.join(default_dir, "true_parameters.json"),
            help="JSON file containing the 'true' model parameters.")
