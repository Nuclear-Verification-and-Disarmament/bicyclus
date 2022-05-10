import util.log as log

import argparse
import os

import cyclus

from copy import deepcopy
from collections import namedtuple

import arviz as az
import numpy as np
import pymc3 as pm
import theano.tensor as tt

import blackbox.blackbox as bb
import blackbox.likelihood as lik

RANDOM_SEED = 12345

# Layour of simulation parameters (mutate())
SimulationParameters = namedtuple('SimulationParameters',
                                  ('u235_frac', 'power_cap'))
# Energy output in MWd
SimulationOutput = namedtuple('SimulationOutput', ('produced_energy', ))


def samples_output_path(i, name='', dir='data', default=None):
    task_id = log.task_identifier()
    if default is None or default == '':
        basepath = os.environ.get('HOME', None)
        default = os.path.join(basepath, dir)
    os.makedirs(default, exist_ok=True)
    return os.path.join(default,
                        'cyclus_trace_{}_{}_{}.cdf'.format(name, task_id, i))


def cyclus_output_path(name='', default=''):
    basepath = os.environ.get('TMP', None)
    out_dir = os.environ.get('CYCLUS_OUT_DIR', '')
    if basepath is None:
        basepath = os.environ.get('WORK', None)
    if basepath is None:
        basepath = default
    os.makedirs(os.path.join(basepath, out_dir), exist_ok=True)
    return os.path.join(basepath, out_dir,
                        name + '_' + log.task_identifier() + '.h5')


class SimpleSampleCyclus(bb.CyclusModel):
    def __init__(self, *args, csk={}, **kwargs):
        super(SimpleSampleCyclus, self).__init__(*args,
                                                 **kwargs,
                                                 cyclus_simstate_kwargs=csk)

    def mutate(self, params):
        log.log_print("mutate:", params)
        self.mut_model = deepcopy(self.model)

        u235_frac = params[0]
        u238_frac = 1 - u235_frac
        powercap = params[1]

        self.mut_model['simulation']['recipe'][1]['nuclide'][0]['comp'] = str(
            u235_frac)
        self.mut_model['simulation']['recipe'][1]['nuclide'][1]['comp'] = str(
            u238_frac)

        self.mut_model['simulation']['facility'][2]['config']['Reactor'][
            'power_cap'] = str(powercap)

        return

    def result(self):
        df = self.last_result.query("TimeSeriesPower")
        log.log_print("Power:", df['Value'].sum())
        return SimulationOutput(
            produced_energy=df['Value'].sum())  #np.array(df['Value'].sum())


class SimpleNormalLikelihood(lik.NormalLikelihood):
    def __init__(self, mu: SimulationOutput, sigma: SimulationOutput):
        self.mu = mu
        self.sigma = sigma

    def log_likelihood(self, so: SimulationOutput):
        loglik = pm.Normal.dist(mu=self.mu.produced_energy,
                                sigma=self.sigma.produced_energy).logp(
                                    so.produced_energy).eval()
        log.log_print(
            "loglik:", loglik,
            "for {} vs {} +/- {}".format(so.produced_energy,
                                         self.mu.produced_energy,
                                         self.sigma.produced_energy))
        return loglik


def main():
    user = os.environ['USER']

    p = argparse.ArgumentParser(description='Cluster sampling test')
    p.add_argument('--index', type=int, default=0, help='Instance index')
    p.add_argument('--run', type=str, help='Name for this run.')
    p.add_argument('--output-path',
                   type=str,
                   default=f'/work/{user}/data/',
                   help='Output path for sample CDF files.')
    p.add_argument('--log-path',
                   type=str,
                   default=f'/work/{user}/job_output/',
                   help='log path.')
    p.add_argument('--samples',
                   type=int,
                   default=400,
                   help='Number of samples per chain per iteration')
    p.add_argument(
        '--iterations',
        type=int,
        default=5,
        help=
        'How many successive iterations with each --samples should be run. Total number of samples is `--samples * --iterations`'
    )
    p.add_argument('--algorithm',
                   type=str,
                   choices=('default', 'metropolis'),
                   help='Which sampling algorithm we should use')
    p.add_argument('--chains', type=int, default=1, help='Number of chains.')
    p.add_argument("--debug", default=False, action="store_true",
                   help="If set, print all output to STDOUT instead of storing"
                        " it in a file. Overrides '--log-path' (if set).")
    args = p.parse_args()
    log.write_to_log_file(run=args.run, outpath=args.log_path,
                          debug=args.debug)

    ssc = SimpleSampleCyclus("sampling0.json",
                             csk={
                                 'output_path':
                                 cyclus_output_path(args.run,
                                                    default=args.output_path)
                             })
    ssc.run_groundtruth()
    ssc.result()
    likelihood = SimpleNormalLikelihood(
        SimulationOutput(produced_energy=55366),
        SimulationOutput(produced_energy=5e3))

    loglikop = lik.CyclusLogLikelihood(likelihood, ssc)

    with pm.Model() as model:
        powercap = pm.Uniform("powercap", 1000, 1300)
        u235frac = pm.Uniform("u235frac", 0.01, 0.1)

        # This is "the inside out" - there is a better way, but that is
        # slightly more complex to implement. Essentially, set
        # observed=groundtruth, and extract powercap, u235frac from the pm
        # model.
        pm.DensityDist(
            "observed",
            lambda v: loglikop(v),
            observed={'v': tt.as_tensor_variable([u235frac, powercap])})

        trace = None
        step = None
        if args.algorithm == 'metropolis':
            step = pm.Metropolis()

        for i in range(0, args.iterations):
            # Refer to https://discourse.pymc.io/t/blackbox-likelihood-example-doesnt-work/5378
            trace = pm.sample(args.samples,
                              tune=int(args.samples / 10),
                              chains=args.chains,
                              return_inferencedata=False,
                              compute_convergence_checks=False,
                              step=step,
                              random_seed=RANDOM_SEED + args.index,
                              trace=trace)
            # this is https://github.com/pymc-devs/pymc-examples/issues/48#issue-839287296
            # see https://arviz-devs.github.io/arviz/api/generated/arviz.from_pymc3.html#arviz-from-pymc3
            #az.from_pymc3(trace, density_dist_obs=False).to_netcdf('/home/vf962887/data/precious_cyclus_trace_{}_{}.cdf'.format(log.task_identifier(), i))
            az.from_pymc3(trace, density_dist_obs=False).to_netcdf(
                samples_output_path(i, args.run, default=args.output_path))

    log.log_print("Success")


if __name__ == "__main__":
    main()
