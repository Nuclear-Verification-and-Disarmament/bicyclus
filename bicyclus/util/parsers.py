""""Predefined parsers to start Bicyclus runs or to analyse its output."""

import argparse
import os
from abc import ABC, abstractmethod
from datetime import datetime


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
            help="PyMC sampling algorithm to be used, e.g., 'Slice' or "
                 "'Metropolis'. See https://www.pymc.io/projects/docs/en/stable/api/samplers.html"
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
            "--tune", type=int, default=100, help="Number of tuning samples")

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


class CyclusRunParser(BaseParser):
    """Parser that can be used to start Cyclus runs."""
    def __init__(self, description="Start Cyclus run.", **kwargs):
        super().__init__(**kwargs)

    def add_args(self):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.parser.add_argument(
            "--name", default=f"Cyclus run {now}", help="Name of Cyclus run")
        self.parser.add_argument(
            "--debug-mode", action="store_true",
            help="If set, run a Cyclus simulation as subprocess, store input "
                 "output, and STDOUT and STDERR in respective files.\n"
                 "Else, dump a JSON Cyclus input file to STDOUT (default).")
        self.parser.add_argument(
            "--infile", type=str, default="input.json",
            help="Name of the Cyclus input file (must be .py, .json or .xml)")
        self.parser.add_argument(
            "--outfile", type=str, default="",
            help="Name of the Cyclus output file, only works in conjunction "
                 "with --debug-mode.")
