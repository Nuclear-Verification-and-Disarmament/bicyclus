import copy
import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from math import log10
from pathlib import Path

from numpy import savetxt
from numpy.random import default_rng
from scipy.stats.qmc import Sobol, discrepancy, scale

from bicyclus.util import log


class CyclusCliModel(ABC):
    """Base class that can mutate a model and run Cyclus simulations.

    This class doesn't use the occasionally brittle python module, but
    instead shells out cyclus. The advantage is also that we can run multiple
    simulations at once.
    """

    def __init__(self):
        self.mutate()
        self.last_sqlite_file = None
        self.result0 = None  # Groundtruth results for later comparison.

    @abstractmethod
    def mutate(self, params=None):
        """Apply a mutation according to params.

        If params is None, a default model should be generated.
        """
        pass

    def simulate(self, cyclus_args={}, timeout=300):
        """Run the Cyclus simulation, storing all data on /tmp.

        Parameters
        ----------
        cyclus_args : dict
            Commandline arguments passed to Cyclus. As default, the input and
            output file names are passed.

        timeout : int
            Maximum (wallclock) duration of one Cyclus simulation in seconds.
            If this time is exceeded, the Cyclus simulation is killed and a
            `TimeoutExpired` exception is raised.
        """
        tmpfile = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
        outfile = os.path.join(tempfile.gettempdir(), tempfile.mktemp() + ".sqlite")

        json.dump(self.mut_model, tmpfile)
        tmpfile.flush()
        tmpfile.close()

        cyclus_args.update({"-i": tmpfile.name, "-o": outfile})
        cyclus_argv = ["cyclus"]
        cyclus_argv.extend([a for l in cyclus_args.items() for a in l])
        log.log_print("Running cyclus:", cyclus_argv)
        this_run = subprocess.run(
            cyclus_argv,
            shell=False,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if this_run.returncode != 0:
            log.log_print(
                "Error calling cyclus. stdout/stderr:", this_run.stderr, this_run.stdout
            )
            raise Exception("Cyclus returned error")
        if self.last_sqlite_file is not None:
            try:
                os.unlink(self.last_sqlite_file)
                self.last_sqlite_file = None
            except Exception as e:
                log.log_print("Error unlinking old SQLite output:", e)

        self.last_sqlite_file = outfile
        log.log_print("Cyclus run finished!")
        os.unlink(tmpfile.name)

        # At this point, result() can extract the desired results from
        # last_sqlite_file.

    @abstractmethod
    def result(self):
        """Extract results from self.last_sqlite_file here."""
        pass

    def run_groundtruth(self):
        self.simulate()
        self.result0 = self.result()
        return self.result0


class CyclusForwardModel(ABC):
    """Class for measurement-independent sampling and uncertainty analysis.

    The parameter file passed to this class uses the same structure as the
    file used by `CyclusCliModel`. As the class uses Sobol sequences to
    efficiently cover the complete input parameter space, the prior
    distribution used must be 'Uniform'.
    """

    def __init__(
        self,
        input_params_fname,
        n_samples_exponent,
        seed,
        data_output_dir=".",
        log_output_dir=".",
        output_fnames="bicyclus_forward",
    ):
        """Initialise class.

        Parameters
        ----------
        input_params_fname : str
            Filename where input parameters and parameter ranges are stored.
            Must be a .json file.

        n_samples_exponent : int
            Logarithm in base 2 of the number of samples, i.e., number of
            samples = 2^n_samples_exponent.

        seed : int
            Seed used to create the Sobol sequences.

        data_output_dir : str or path, optional
            Directory where Cyclus output files are stored. Default is current
            directory.

        log_output_dir : str or path, optional
            Directory where parameter samples and log are stored. Default is
            current directory.

        output_fnames : str, optional
            Basename of all output files, i.e., file with drawn parameters,
            log, and Cyclus .sqlite files.
        """
        self.rng = default_rng(seed=seed)
        log.log_print(f"Using RNG {self.rng} with seed {seed}")

        self.input_params_fname = input_params_fname
        self.data_output_dir = Path(data_output_dir).absolute()
        self.log_output_dir = Path(log_output_dir).absolute()
        self.output_fnames = output_fnames
        # If not existent, create data and log directories.
        for dir_ in (self.data_output_dir, self.log_output_dir):
            dir_.mkdir(mode=0o760, parents=True, exist_ok=True)

        self.n_samples_exponent = n_samples_exponent
        self.parameter_names, self.samples = self.generate_samples()

    def generate_samples(self):
        """Create samples using Sobol sequences, a QMC method."""
        with open(self.input_params_fname, "r") as f:
            parameters = json.load(f)
        parameter_names = list(parameters.keys())
        log.log_print(
            "Generating samples for the following "
            f"{len(parameter_names)} parameters: "
            f"{', '.join(parameter_names)}"
        )

        sobol_sampler = Sobol(d=len(parameter_names), seed=self.rng)
        lower_bounds = []
        upper_bounds = []
        for param in parameters.values():
            if param["type"] != "Uniform":
                msg = (
                    "At the moment, parameter distributions must be Uniform"
                    " distributions."
                )
                raise ValueError(msg)
            lower_bounds.append(param["lower"])
            upper_bounds.append(param["upper"])

        unscaled_samples = sobol_sampler.random_base2(self.n_samples_exponent)
        scaled_samples = scale(unscaled_samples, lower_bounds, upper_bounds)
        discrepancy_ = discrepancy(unscaled_samples)
        log.log_print(
            f"Generated {scaled_samples.shape[0]} samples for "
            f"{scaled_samples.shape[1]} parameters with discrepancy "
            f"{discrepancy_}."
        )
        output_fname = self.log_output_dir / f"parameters_{self.output_fnames}.csv"
        savetxt(output_fname, scaled_samples, header=" ".join(parameter_names))
        log.log_print("Stored samples in:", output_fname)

        return parameter_names, scaled_samples

    @abstractmethod
    def generate_input_file(self, sample):
        """Generate a Cyclus input file using the form described below.

        Parameters
        ----------
        sample : dict
            Dict with keys being the parameter names (as defined in the input
            parameter file) and values being the drawn parameter samples.

        Returns
        -------
        dict
            A dict containing the Cyclus input file. This will be converted
            and stored as a .json file.
        """
        pass

    def run_simulations(self, cyclus_args={}, const_sim_params={}, timeout=300):
        """Run Cyclus simulations using generated set of parameters.

        Parameters
        ----------
        cyclus_args : dict, optional
            Commandline args passed to Cyclus.

        const_sim_params : dict, optional
            Parameters that are not sampled but should be passed to the input
            file generator in addition to the sampled parameters.

        timeout : int
            Maximum (wallclock) duration of one Cyclus simulation in seconds.
            If this time is exceeded, the Cyclus simulation is killed and a
            `TimeoutExpired` exception is raised.
        """
        n_samples = 2**self.n_samples_exponent
        width_samples = int(log10(2**self.n_samples_exponent)) + 1

        log.log_print(
            "Starting simulations. Storing Cyclus output files in:"
            f"{self.data_output_dir}"
        )

        for i, sample in enumerate(self.samples):
            log.log_print("Generating input file.")
            parameters = {k: v for k, v in zip(self.parameter_names, sample)}
            if not const_sim_params.keys().isdisjoint(parameters.keys()):
                intersection = const_sim_params.keys() & parameters.keys()
                msg = (
                    "Sampled parameters and constant parameters must "
                    f"differ. Common parameters: {intersection}"
                )
                raise RuntimeError(msg)

            parameters.update(const_sim_params)
            input_file = self.generate_input_file(sample=parameters)

            tmpfile = tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".json"
            )
            json.dump(input_file, tmpfile)
            tmpfile.flush()
            tmpfile.close()

            outfile = self.data_output_dir / f"cyclus_{self.output_fnames}_{i}.sqlite"

            # Existing files are deleted, else Cyclus would append results.
            if os.path.isfile(outfile):
                os.unlink(outfile)

            cyclus_argv = ["cyclus", "-i", tmpfile.name, "-o", outfile]
            cyclus_argv.extend([j for i in cyclus_args.items() for j in i])
            log.log_print("Running Cyclus:", cyclus_argv)

            run = subprocess.run(
                cyclus_argv,
                shell=False,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if run.returncode != 0:
                log.log_print(
                    "Error running cyclus. stdout/stderr:", run.stdout, run.stderr
                )
                raise RuntimeError("Cyclus returned error")

            log.log_print(
                f"Simulation finished. {i + 1:>{width_samples}} / {n_samples}"
            )
            os.unlink(tmpfile.name)

        log.log_print(f"Finished sampling of {n_samples} samples.")
