import numpyro 
import numpyro.distributions as dist
import jax
import jax.numpy as jnp 
import optax
import sys, os
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd, '../'))


N_chains = 4
N_steps = 100

#   *---*   Automatic from here  *---*   #
from data.data_classes import Pantheon
pantheon = Pantheon()

from source.models import cpl_single
model = cpl_single
nuts_kernel = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=100, num_samples=N_steps, num_chains=N_chains, chain_method='vectorized')

mcmc.run(jax.random.PRNGKey(3141), pantheon.data)

# Save output
import pickle
filename_base = os.path.split(__file__)[-1][:-3]
file_idx = 0
while os.path.isfile(os.path.join(pwd, f'../output/{filename_base}_{file_idx}.pickle')):
    file_idx += 1
output_name = os.path.join(pwd, f'../output/{filename_base}_{file_idx}.pickle')

# In principle, we can save params during the loop above to track the evolution of the proposal
output_samples = mcmc.get_samples()
with open(output_name, 'wb') as handle:
    pickle.dump(output_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


