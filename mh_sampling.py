from __future__ import print_function
import numpy as np

from scipy.stats import multivariate_normal as smn
import matplotlib.pyplot as plt
import sys
x = np.array([4,4])
cov = np.array([[1,0.8],[0.8,1]])
z_t = [np.random.normal(0,10), np.random.normal(0,10)]
max_iter = 1000
count = 0
accept_count = 0
log = np.zeros([max_iter]+list(x.shape))
sigma = float(sys.argv[1])
var = sigma**2
for i in range(max_iter):
	z_u = np.random.normal(z_t,var)
	val = (smn.logpdf(z_t, x, cov) + smn.logpdf(z_t, z_u, var)) - (smn.logpdf(z_u,x,cov) + smn.logpdf(z_u, z_t, var))
	val = min(1., np.exp(val))
	print(val, end='\r')
	u = np.random.uniform()
	if u < val : 
		z_t = z_u
		log[accept_count] = z_t
		accept_count +=1
	count += 1
print("\n{} : {}".format(accept_count, count))
for i in range(max_iter):
	plt.scatter(log[i][0], log[i][1])
plt.show()
