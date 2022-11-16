import numpyro 
import numpyro.distributions as dist 
import jax.numpy as jnp
from source.distributions import MixtureMultivariateGaussian

class MixtureMultivariateGuide():
    def __init__(self, model_name, N_gauss, initial_locs, initial_scales=None):
        # N_gauss could be inferred from initial_locs, but this is more readable
        self.model_name = model_name
        self.N_gauss = N_gauss
        self.initial_locs = jnp.array(list(initial_locs.values()))
        if not initial_scales:
            self.initial_scales_tril = jnp.stack([0.05*jnp.eye(len(list(initial_locs))) for k in range(self.N_gauss)])

    def guide(self, data):
        weights = numpyro.param("weights", jnp.ones(self.N_gauss)/self.N_gauss)
        locs = numpyro.param("locs", jnp.broadcast_to(jnp.array(self.initial_locs), (self.N_gauss, self.initial_locs.shape[0])))
        scale_tril = numpyro.param("scale_tril", self.initial_scales_tril, constraint=dist.constraints.positive)
    
        params = numpyro.sample(self.model_name, MixtureMultivariateGaussian(weights=weights, locs=locs, scales_tril=scale_tril))
