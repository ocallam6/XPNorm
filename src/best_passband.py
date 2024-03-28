'''
Here we want to find the bast passbands for inferring the extinction

'''
import numpy as np

# sample i from S=(1,...,n)
# set S=(1,...,n)-{i}
# while lenght set S>0
# sample m from range(1,len(S))
# sample (j_1,...,j_n) from S and remove each term from S
# sample a simplex of length S, these are the coefficients for the $j_1,...,j_m$ extinction
#end while
#compute loss

# Gaia G U(0.7,0.95)
# Gaia BP U(0.97,1.28)
# Gaia RP U(0.55,0.69)
# divide the last 3 by 3.1
# 2mass J r(a) U(0.71,0.73)
# 2mass H U(0.45,0.47)
# 2mass ks U(0.34,0.36)

s=[0,1,2,3,4,5]



import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

# Generate some synthetic data
error=jnp.array([0.00308123, 0.01036897, 0.0043106 , 0.03507253, 0.03733693,
       0.05396977])

mu_error=0.086
# Define the Bayesian model
def model(err,mu_err):
    # Priors for the elements of the matrix
    g = numpyro.sample("g", dist.Uniform(0.7, 0.95))
    bp = numpyro.sample("bp", dist.Uniform(0.97,1.28))
    rp = numpyro.sample("rp", dist.Normal(0.55, 0.69))
    j = numpyro.sample("j", dist.Uniform(0.71, 0.73))
    h = numpyro.sample("h", dist.Uniform(0.45,0.47))
    ks = numpyro.sample("ks", dist.Normal(0.34, 0.36))

    extinction_vector=jnp.array([g,bp,rp,j,h,ks])

    length_matrix = int(6*6)

    with numpyro.plate("plate_i", length_matrix):
       coeff = numpyro.sample("coeff", dist.Uniform(-1.0,1.0))

    coeff=coeff.reshape(6,6)
    row_sum=coeff.sum(-1)
    det=jnp.linalg.det(coeff)

    magnitude=coeff@(extinction_vector/err)
    mu_term=row_sum*(extinction_vector/mu_err)

    with numpyro.plate("data", 1):
        numpyro.sample("obs", dist.Normal((magnitude**2).sum()+(mu_term**2).sum(), 1/det), obs=0.0)


nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=2000, num_warmup=2000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key,error,mu_error)

posterior_samples = mcmc.get_samples()






