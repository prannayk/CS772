from __future__ import print_function
import numpy as np
import scipy as scipy
import scipy.stats as stats
import thread
from load import loader

t = loader("20newsgroups.mat")
word_list = []
for word in t["words"]:
	word_list.append(str(word[0][0]))
print(len(word_list))
matrix = t["X"]
N = matrix.shape[0]
M = matrix.shape[1]
K = 20
print("The dimensions of U are : {} x {}".format(N, K))
print("The dimensions of V are : {} x {}".format(M, K))
a_u = 10.
b_u = 120.
a_v = 10.
b_v = 80.
U = np.random.gamma(a_u, b_u, [N, K])
V = np.random.gamma(a_v, b_v, [M,K])
num_iterations = 10
X_out = np.zeros([N,M, K])

def sample(i, name):
	print("Sampling for row : {}".format(i+1))
	for j in range(M):
		P_vec = np.zeros([K])
		for k in range(K):
			P_vec[k] = U[i,k]*V[j,k]
		P_vec = P_vec / np.sum(P_vec)
		X_out[i,j] = np.random.multinomial(matrix[i,j], P_vec).astype(np.float32)

for i in range(num_iterations):
	print("Running iteration : {}".format(i + 1))
	U_new = np.zeros_like(U)
	V_new = np.zeros_like(V)
	print("Sampling from multinomial")
	for i in range(N):
		thread.start_new_thread(sample, (i,str(i)))
	thread.join()
	print("sampling from gamma for u")
	for i in range(N):
		a_i = np.sum(X_out, axis=1) + a_u
		b_i = np.sum(V, axis=0) + np.ones([K])*(1/b_u)
		for k in range(K):
			U_new[i, k] = np.random.gamma(a_i[k], b_i[k])
	print("sampling from gamma for v")
	for j in range(M):
		a_i = np.sum(X_out, axis=1) + a_v
		b_i = np.sum(V, axis=0) + np.ones([K])*(1/b_v)
		for k in range(K):
			V_new[j, k] = np.random.gamma(a_i[k], b_i[k])
	print("computing error and seeing results")
	error = np.sum(matrix-X_new)
	print("Error is : {}".format(error))
