from __future__ import print_function
import numpy as np
import scipy as scipy
import scipy.stats as stats
from scipy.sparse import dok_matrix
from load import loader
import time

t = loader("20newsgroups.mat")
word_list = []
for word in t["words"]:
	word_list.append(str(word[0][0]))
matrix = np.array(t["X"][:5000].todense())
sparse = dok_matrix(t["X"][:5000])
keys = sparse.keys()
train_size = 8 * len(keys) / 10
print("Train size: {} out of : {}".format(train_size, len(keys)))
train_keys = keys[:train_size]
test_keys = list(set(keys) - set(train_keys))
N = matrix.shape[0]
M = matrix.shape[1]
K = 20
print("Read to process:")
print("The dimensions of U are : {} x {}".format(N, K))
print("The dimensions of V are : {} x {}".format(M, K))
a_u = 10.
b_u = 120.
a_v = 10.
b_v = 80.
U = np.random.gamma(a_u, 1./b_u, [N, K])
V = np.random.gamma(a_v, 1./b_v, [M,K])
num_iterations = 1000
X_out = np.zeros([N,M, K])
sparse_hat = dok_matrix(sparse, dtype=np.int32)
flag = True
for iteration in range(num_iterations):
	start_iter = time.time()
	X_out = np.zeros([N,M, K])
	U_new = np.zeros_like(U)
	V_new = np.zeros_like(V)
	for key in keys:
			print("Sampling for row : {} and column : {}".format(key[0], key[1]), end='\r')
			P_vec = np.zeros([K])
			for k in range(K):
				P_vec[k] = U[key[0],k]*V[key[1],k]
			P_vec = P_vec / np.sum(P_vec)
			X_out[key[0],key[1]] = np.random.multinomial(matrix[key[0],key[1]], P_vec).astype(np.float32)
	a_i = np.sum(X_out, axis=1) + a_u
	b_i = np.sum(V, axis=0, keepdims=True) + (b_u)
	U_new = np.random.gamma(a_i, 1./b_i)
	a_i = np.sum(X_out, axis=0) + a_v
	b_i = np.sum(U_new, axis=0, keepdims=True) + b_v
	V_new = np.random.gamma(a_i,1./b_i)
	calc = np.matmul(U_new, np.transpose(V_new))
	X_new = np.zeros_like(calc)
	error_train = 0.
	error_test = 0.
	for key in train_keys :
		print("Doing for key : {:7d}".format(key[0]), end='\r')
		X_new[key[0], key[1]] = np.random.poisson(calc[key[0], key[1]])
		error_train += np.abs(X_new[key[0], key[1]] - matrix[key[0], key[1]])
	for key in test_keys :
		print("Doing for key : {:7d}".format(key[0]), end='\r')
		X_new[key[0], key[1]] = np.random.poisson(calc[key[0], key[1]])
		error_test += np.abs(X_new[key[0], key[1]] - matrix[key[0], key[1]])
	error_train /= len(train_keys)
	error_test /= len(test_keys)
	if flag or iteration < 50 :
		flag = False
		X_mc = np.copy(X_new)
	else :
		X_mc = ((float(i) / i+1)*X_mc + (1./i+1)*X_new)
	count = 0
	U = np.copy(U_new)
	V = np.copy(V_new)
	print("Errors for iteration ({}) with test {} and train {} in time ({})".format(iteration+1,error_test, error_train, time.time() - start_iter))
