import copy
import json
import os
import subprocess
import tempfile

from ..util import log


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
        tmpfile.flush()
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


# This class is deprecated
class CyclusModel:
    def __init__(self, filename, cyclus_simstate_kwargs={}):
        msg = ("This class is deprecated. Please use 'CyclusCliModel', which "
               "has the added benefit of running on multiple cores.")
        raise DeprecationWarning(msg)
