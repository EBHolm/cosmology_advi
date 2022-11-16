import jax.numpy as jnp
import jax 
import numpyro
import numpyro.distributions as dist 

class HypercubeConstraint(dist.constraints.Constraint):
    # Custom constraint used for defining the support of MultivariateUniform below
    # https://num.pyro.ai/en/stable/_modules/numpyro/distributions/constraints.html
    def __init__(self, boundaries):
        self.boundaries = boundaries
        
    def __call__(self, x):
        # Return whether the vector x is inside of the support
        is_inside_support = True
        for idx_boundary, bound in self.boundaries:
            is_inside_support *= (x[..., idx_boundary] >= bound[0]) & (x[..., idx_boundary] <= bound[1])
        return is_inside_support
    
    def feasible_like(self, prototype):
        # Return random possible value within support of given shape
        return jax.numpy.broadcast_to(
            (self.lower_bound + self.upper_bound) / 2, jax.numpy.shape(prototype)
        )

class MultivariateUniform(dist.Distribution):
    # https://num.pyro.ai/en/stable/_modules/numpyro/distributions/distribution.html#Distribution
    def __init__(self, boundaries, validate_args=None):
        
        super().__init__(batch_shape=jnp.shape(boundaries)[:-2], event_shape=(jnp.shape(boundaries)[-2],), validate_args=validate_args)

        self.boundaries = boundaries
        self.support = HypercubeConstraint(boundaries)
        self.N_dists = jnp.shape(boundaries)[-2]
        self.dists = []
        self.logprob = 0.0
        for bound in boundaries:
            self.dists.append(dist.Uniform(bound[0], bound[1]))
            self.logprob += -jnp.log(bound[1] - bound[0])

    def sample(self, key, sample_shape=(3,)): # NB: Hardcoded 3 uniforms here!
        out = jnp.zeros(sample_shape)
        for idx_sample, dist in enumerate(self.dists):
            out = out.at[idx_sample].set(dist.sample(key))
        return out 

    def log_prob(self, value):
        # Assume already within support 
        return self.logprob

class MixtureMultivariateGaussian(dist.Distribution):
    support = dist.constraints.real_vector

    def __init__(self, weights, locs, scales_tril, validate_args=None):
        self.N_normals = jnp.shape(scales_tril)[0]
        self._gaussians = []
        for loc, scale_tril in zip(locs, scales_tril):
            self._gaussians.append(dist.MultivariateNormal(loc=loc, scale_tril=scale_tril))

        batch_shape = jnp.shape(scales_tril)[:-3]
        event_shape = jnp.shape(scales_tril)[-1:]
        self.normalization = jnp.sum(weights)
        self.weights = weights/self.normalization
        self._categorical = dist.Categorical(weights)

        super(MixtureMultivariateGaussian, self).__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
        
    def sample(self, key, sample_shape=()):
        key, key_idx = jax.random.split(key)
        samples = jnp.array([self._gaussians[idx].sample(key, sample_shape) for idx in range(self.N_normals)])
        ind = self._categorical.sample(key_idx, sample_shape)
        return samples[..., ind, :]
    
    def log_prob(self, value):
        probs_mixture = jnp.array([self._gaussians[idx].log_prob(value[..., :]) for idx in range(self.N_normals)])
        weighted_probs = self.weights*jnp.exp(probs_mixture)
        return jnp.log(jnp.sum(weighted_probs, axis=-1))
        