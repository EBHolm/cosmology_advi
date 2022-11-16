import jax.numpy as jnp 
import numpy as np 
import os

class Union2:
    '''data class for the union 2.1 data. The columns 2-5 are used
    which contain redshift, distance modulus and distance modulus error
    as well as the probability of the host galaxy to be of high mass.
    But this property is not important. The covariance is imported 
    and directly inverted.'''
    def __init__(self):
        name = os.path.join(os.path.dirname(__file__), 'union2/union2.txt')
        cov = os.path.join(os.path.dirname(__file__), 'union2/union2_covsys.txt')
        matrix1 = np.genfromtxt(name, delimiter='\t', skip_header=5, usecols=(1,2,3,4))
        matrix = jnp.asarray(matrix1)
        self.redshift = matrix[:,0]
        self.dm = matrix[:,1]
        self.err = matrix[:,2]
        self.prob = matrix[:,3]
        
        covariance = np.genfromtxt(cov, delimiter='\t', usecols=(range(len(self.redshift))))
        self.cov = covariance
        self.inv_cov = jnp.asarray(np.linalg.inv(covariance))

class Pantheon:
    '''data class for pantheon data. Only four columns are used.
    The covarince matrix is passed as a list of size 1048x1048
    and can be reshaped as a matrix.'''
    def __init__(self):
        name = os.path.join(os.path.dirname(__file__), 'pantheon/pantheon.txt')
        cov = os.path.join(os.path.dirname(__file__), 'pantheon/pantheon_covsys.txt')
        matrix1 = np.genfromtxt(name, delimiter='', skip_header=7)
        matrix = jnp.asarray(matrix1)
        self.redshift1 = matrix[:,7]
        self.redshift2 = matrix[:,9]
        self.dm = matrix[:,40]
        self.err =matrix[:,42]
        covariance = np.loadtxt(cov)
        cov_shape = int(covariance[0])
        cov = covariance[1:].reshape((cov_shape, cov_shape)) + np.diag(self.err**2)
        self.cov = cov 
        self.inv_cov = jnp.asarray(np.linalg.inv(cov))

        self.data = {}
        self.data['z'] = self.redshift1
        self.data['distance_modulus'] = self.dm
        self.data['cov'] = self.cov