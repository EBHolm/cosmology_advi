import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp 
from source.cosmology import distance_modulus
from source.distributions import MultivariateUniform, MixtureMultivariateGaussian

prior_bounds = jnp.array([[0.0, 1.0], [-10.0, 10.0], [-10.0, 10.0]])
Omega_m_prior = dist.Uniform(*prior_bounds[0])
w0_prior = dist.Uniform(*prior_bounds[1])
wa_prior = dist.Uniform(*prior_bounds[2])

def cpl_single(data):
    Omega_m = numpyro.sample("Omega_m", Omega_m_prior)
    w0 = numpyro.sample("w0", w0_prior)
    wa = numpyro.sample("wa", wa_prior)
    theta_dict = {"Omega_m":Omega_m, "w0":w0, "wa":wa}
    mu = distance_modulus(theta_dict, data['z'])
    numpyro.sample("y", dist.MultivariateNormal(mu, data['cov']), obs=data['distance_modulus'])

def cpl_vec(data):
    cpl_params = numpyro.sample("cpl_params", MultivariateUniform(prior_bounds))
    theta_dict = {"Omega_m": cpl_params[0], "w0": cpl_params[1], "wa": cpl_params[2]}
    mu = distance_modulus(theta_dict, data['z'])
    numpyro.sample("y", dist.MultivariateNormal(mu, data['cov']), obs=data['distance_modulus'])