import math
import traceback

import numpy as np
import aesara.tensor as at


class LikelihoodFunction:
    def __init__(self):
        pass

    def log_likelihood(self, X):
        assert False, "LikelihoodFunction.log_likelihood needs to be overridden"

    def __call__(self, X):
        return self.log_likelihood(X)


class NormalLikelihood(LikelihoodFunction):
    def __init__(self, mus, sigmas):
        assert len(mus) == len(sigmas)
        self.mus = mus
        self.sigmas = sigmas

    def log_likelihood(self, X):
        def normal_pdf(x, mu=0, sigma=1):
            """Normal distribution with parameters mu, sigma evaluated at x."""
            return (
                math.exp(-((x - mu) ** 2) / (2 * sigma**2))
                / (2 * math.pi) ** 0.5
                / sigma
            )

        assert len(X) == len(self.mus)

        logsum = sum(
            math.log(normal_pdf(X[i], mu=self.mus[i], sigma=self.sigmas[i]))
            for i in range(0, len(X))
        )

        return logsum


def simplegrad(func, theta, releps=1e-2, reltol=1e-2, scale_eps=1 / 2, maxiter=5):
    """Calculate simple numeric gradient of func at theta.

    func should take list/array theta as only argument.
    """
    grad = np.zeros_like(theta, dtype=np.float)
    upper, lower = np.copy(theta), np.copy(theta)
    totalcount = 0
    for i, t in enumerate(theta):
        count, flipflop = 0, 0
        lastdiff = 0
        this_eps = releps * t

        while True:
            upper[i] += this_eps / 2
            lower[i] -= this_eps / 2
            diff = (func(upper) - func(lower)) / this_eps
            print(f"diff = {diff}, eps = {this_eps}, {upper} / {lower}, {lastdiff}")
            count += 1
            totalcount += 1
            if count > 5:
                grad[i] = diff
                break

            upper[i] -= this_eps / 2
            lower[i] += this_eps / 2

            ratio = lastdiff / diff
            if ratio - 1 >= 0:
                if abs(ratio - 1) < reltol:
                    grad[i] = diff
                    break
                else:
                    lastdiff = diff
            else:
                flipflop += 1
                lastdiff = diff

            if count >= maxiter or flipflop > 4:  # Arbitrary parameters so far
                grad[i] = diff
                break

            this_eps *= scale_eps
    print(f"Grad: {len(theta)} dimensions took {totalcount} iterations")
    return grad


class CyclusLogLikelihood(at.Op):
    """Calculate the likelihood of the simulation result using a Aesara Tensor.

    See https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html#blackbox_external_likelihood_numpy
    """

    itypes = [at.dvector]
    otypes = [at.dscalar]

    def __init__(self, likelihood_callable, cyclus_model, maxiter=5, memoize=False):
        """Create a CylusLogLikelihood object.

        Parameters
        ----------
        likelihood_callable : LikelihoodFunction (or derivative thereof)
            The object that extracts simulation results and calculates the
            loglikelihood.
        cyclus_model : blackbox.CyclusCliModel (or derivative thereof)
            The Cyclus model.
        maxiter : int
            TODO finish docstring
        memoize : bool
            Store intermediate results in a lookup-table and use these if
            possible to speed up calculations.
            TODO finish docstring
        """
        self.likelihood = likelihood_callable
        self.cyclus_model = cyclus_model
        self.loggrad = LogLikelihoodGrad(
            likelihood_callable, cyclus_model, maxiter=maxiter
        )
        self.memo = {} if memoize else None

    def perform(self, node, inputs, outputs):
        """Mutate and run Cyclus model, then calculate likelihood.

        Calculates the log likelihood of the current random variable vector
        compared to the ground truth.

        The comparison will assume a normal distribution (or uniform
        distribution, or whatever) with mean at the given result. The
        likelihood is then the calculated probability of each ground truth
        result value with respect to the current value's distribution.
        """
        (params,) = inputs

        # likelihood calculation is deterministic
        if self.memo is not None and str(params) in self.memo:
            print("(memoized)")
            outputs[0][0] = self.memo[str(params)]
            return

        try:
            self.cyclus_model.mutate(params)
            self.cyclus_model.simulate()
            current = self.cyclus_model.result()
            loglik = self.likelihood(current)
        except Exception as e:
            msg = (
                f"Cyclus Model {self.cyclus_model} had error: {e}; "
                "returning likelihood = -np.inf\n"
                f"Traceback:\n",
                traceback.format_exc(),
            )
            print(msg)
            loglik = -np.inf

        if self.memo is not None:
            self.memo[str(params)] = np.array(loglik)

        outputs[0][0] = np.array(loglik)

    def grad(self, inputs, g):
        (theta,) = inputs
        return [g[0] * self.loggrad(theta)]


class LogLikelihoodGrad(at.Op):
    """Gradient Op for a nuclear fuel cycle model."""

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, loglike, cyclus_model, maxiter):
        self.loglike = loglike
        self.cyclus_model = cyclus_model
        self.maxiter = maxiter

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        def differentiable(args):
            self.cyclus_model.mutate(args)
            self.cyclus_model.simulate()
            r = self.cyclus_model.result()
            return self.loglike(r)

        grads = simplegrad(differentiable, theta, maxiter=self.maxiter)
        outputs[0][0] = grads
