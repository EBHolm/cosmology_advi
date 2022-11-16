import numpyro 
import numpyro.distributions as dist
import jax
import jax.numpy as jnp 
import optax
import sys, os
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd, '../'))

"""
    Implementation of notebooks/GaussianMixtureGuide.ipynb using source files.

"""
N_chains = 4
N_steps = 2000
data_size = 2000

true_param = {
    'h':0.7,
    'Omega_m':0.29,
    'w0':-1.02,
    'wa':0.01
}

#   *---*   Automatic from here  *---*   #
# Generate mock data 
from source.cosmology import distance_modulus
key = jax.random.PRNGKey(3141)
key, subkey, subkey2, subkey3 = jax.random.split(key, 4)
shape = (data_size,)
z_random = jax.random.uniform(key, shape, minval=0.01, maxval=1.5)
z = jnp.sort(z_random)
sigma = 0.3 * jnp.log(1+z)
dist_mod_err = sigma * jax.random.normal(subkey, shape) 
cov = jnp.diag(sigma**2)
dist_mod_data = distance_modulus(true_param, z) #+ dist_mod_err
data = {
    'distance_modulus': dist_mod_data,
    'cov': cov,
    'z': z
}

# Define model and guide 
from source.models import cpl_single
model = cpl_single
nuts_kernel = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=100, num_samples=N_steps)

mcmc.run(jax.random.PRNGKey(3141), data, num_chains=N_chains, chain_method='vectorized')