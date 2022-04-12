import sqlite3

import blackbox.blackbox as bb

from models.srs import cyclus_input
from cyclus_db import extract


class SRSDemoModel(bb.CyclusCliModel):
    def mutate(self, params=None):
        if params is None:
            self.mut_model = cyclus_input.simulation()
            return
        model_parameters = {}  # To Do: create these from the params vector.
        self.mut_model = cyclus_input.simulation(parameters=model_parameters)

    def result(self):
        filename = self.last_sqlite_file
        print(
            extract.run_with_conn(filename,
                                  extract.extract_isotope_concentrations,
                                  dict(agent_name='PlutoniumSink')))


def main():
    cdm = SRSDemoModel()

    cdm.simulate()
    cdm.result()
    cdm.simulate()


if __name__ == "__main__":
    main()
