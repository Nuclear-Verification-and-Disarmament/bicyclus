import copy
import json
import os
import subprocess
import tempfile

# Both cyclus imports can be removed if the CyclusModel class gets removed.
import cyclus.simstate as cycsim
import cyclus.memback as cycmb

from util import log


class CyclusCliModel:
    """Base class that can mutate a model and run Cyclus simulations.

    This class doesn't use the occasionally brittle python module, but
    instead shells out cyclus. The advantage is also that we can run multiple
    simulations at once.
    """
    def __init__(self):
        self.mutate()
        self.last_sqlite_file = None
        self.result0 = None  # Groundtruth results for later comparison.

    def mutate(self, params=None):
        """Apply a mutation according to params.

        If params is None, a default model should be generated.
        """
        self.mut_model = copy.deepcopy(self.model)
        # override to do actual mutation!

    def simulate(self, cyclus_args={}):
        tmpfile = tempfile.NamedTemporaryFile(delete=False,
                                              mode='w',
                                              suffix='.json')
        outfile = os.path.join(tempfile.gettempdir(),
                               tempfile.mktemp() + '.sqlite')

        json.dump(self.mut_model, tmpfile)
        tmpfile.flush()  # buffering -1, my ass
        tmpfile.close()

        cyclus_args.update({'-i': tmpfile.name, '-o': outfile})
        cyclus_argv = ['cyclus']
        cyclus_argv.extend([a for l in cyclus_args.items() for a in l])
        log.log_print('Running cyclus:', cyclus_argv)
        this_run = subprocess.run(cyclus_argv,
                                  shell=False,
                                  timeout=300,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        if this_run.returncode != 0:
            log.log_print('Error calling cyclus. stdout/stderr:',
                          this_run.stderr, this_run.stdout)
            raise Exception('Cyclus returned error')
        if self.last_sqlite_file is not None:
            try:
                os.unlink(self.last_sqlite_file)
                self.last_sqlite_file = None
            except Exception as e:
                log.log_print('Error unlinking old SQLite output:', e)

        self.last_sqlite_file = outfile
        log.log_print('Cyclus run finished!')
        os.unlink(tmpfile.name)

        # At this point, result() can extract the desired results from
        # last_sqlite_file.

    def result(self):
        """Extract results from self.last_sqlite_file here."""
        return None

    def run_groundtruth(self):
        self.simulate()
        self.result0 = self.result()
        return self.result0


# Note: This model only works on a single core (chains=1 or cores=1 in PyMC).
# For parallelizing, we will have to create several identical copies.
class CyclusModel:
    def __init__(self, filename, cyclus_simstate_kwargs={}):
        msg = ("This class will probably be deprecated. It was originally "
               "created to use Cyclus' Python API, however the implementation "
               "in 'CyclusCliModel' works fine (and runs on multiple cores).")
        raise PendingDeprecationWarning(msg)

        self.model = CyclusModel.load_model(filename)
        self.mut_model = self.model
        self.cyclus_simstate_kwargs = cyclus_simstate_kwargs
        self.cyclus_output_path = cyclus_simstate_kwargs.get(
            'output_path', None)
        self.last_result = None

    def load_model(filename):
        with open(filename, 'r') as f:
            model = json.load(f)
            return model

    def mutate(self, params=None):
        """Override this to update the internal `mut_model` used for the
        simulation. `params` is a parameter vector sampled by PyMC."""
        self.mut_model = copy.deepcopy(self.model)
        # do mutation in your overridden method.

    def simulate(self):
        """Runs simulation with current model, and returns a cyclus MemBack
        (and updates the internal result)."""
        # Remove previous SQLite/HDF5 file.
        try:
            # possible problem: too many open file descriptors when using SQLite!
            # Use HDF5 to not overflow FDs.
            if self.cyclus_output_path:
                os.remove(self.cyclus_output_path)
        except OSError:
            pass

        js = json.dumps(self.mut_model)
        memback = cycmb.MemBack()
        simst = cycsim.SimState(input_file=js,
                                input_format="json",
                                memory_backend=memback,
                                **self.cyclus_simstate_kwargs)
        simst.load()
        simst.run()

        if self.last_result is not None:
            self.last_result.close()
        self.last_result = memback
        return memback

    def result(self):
        """Override this method to extract a RV vector (result vector)
        from the self.result memback database. The resulting vector
        is the foundation for PyMC sampling, and will be compared to the
        initial results obtained within `run_groundtruth()`, and are
        the input to the selected likelihood function class (module
        `likelihood`).

        The MemBack database can be queried like this:
            >>> result.query("Recipes", [("Recipe", "==", "natu")])
                                              SimId Recipe  QualId
            0  45999ff7-0cc4-4a16-bb93-acf6bdf40ba5   natu       1

        """
        return None

    def run_groundtruth(self):
        """Run a simulation with current parameters, and store the result
        vector returned by `self.result()` as "ground truth", to which later
        simulations will be compared to.
        """
        self.simulate()
        self.result0 = self.result()

    def get_groundtruth_current_result(self):
        return (self.result0, self.result())
