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

N_steps = 100
data_size = 2000

true_param = {
    'h':0.7,
    'Omega_m':0.29,
    'w0':-1.02,
    'wa':0.01
}

initial_values = {
    'Omega_m': 0.25,
    'w0': -0.8,
    'wa': 0.01
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
from source.guides import MixtureMultivariateGuide
from source.models import cpl_vec
N_gauss = 2
learning_rate = 0.001
model = cpl_vec
guide = MixtureMultivariateGuide("cpl_params", N_gauss, initial_values)

# Run inference
svi_multi = numpyro.infer.SVI(model,
    guide.guide,
    optax.adam(learning_rate),
    numpyro.infer.Trace_ELBO()
)

state, loss = svi_multi.stable_update(svi_multi.init(jax.random.PRNGKey(3141), data), data)
for idx_step in range(N_steps):
    state, loss = svi_multi.update(state, data)
    if idx_step % 25 == 0:
        print(f"Step {idx_step}: loss={loss}")

# Save output
import pickle
filename_base = os.path.split(__file__)[-1][:-3]
file_idx = 0
while os.path.isfile(os.path.join(pwd, f'../output/{filename_base}_{file_idx}.pickle')):
    file_idx += 1
output_name = os.path.join(pwd, f'../output/{filename_base}_{file_idx}.pickle')

# In principle, we can save params during the loop above to track the evolution of the proposal
output_params = svi_multi.get_params(state)
with open(output_name, 'wb') as handle:
    pickle.dump(output_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
