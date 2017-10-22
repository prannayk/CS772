from __future__ import print_function
import numpy as np
import scipy as scipy
import scipy.stats as stats
from scipy.sparse import dok_matrix
import threading
from threading import Thread
from load import loader
import time
t = loader("20newsgroups.mat")
word_list = []
for word in t["words"]:
	word_list.append(str(word[0][0]))
print(len(word_list))
matrix = np.array(t["X"].toarray())
sparse = dok_matrix(t["X"])
keys = sparse.keys()
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
num_iterations = 1000
X_out = np.zeros([N,M, K])
U_new = np.zeros_like(U)
V_new = np.zeros_like(V)
count = 0
def sample(i,j, name):
	X_out = np.zeros([N,M, K])
	print("Sampling for entry : {},{}".format(i+1,j+1),end='\r')
	P_vec = np.zeros([K])
	for k in range(K):
		P_vec[k] = U[i,k]*V[j,k]
	P_vec = P_vec / np.sum(P_vec)
	X_out[i,j] = np.random.multinomial(matrix[i,j], P_vec).astype(np.float32)
	print("Sampling for entry : {},{} completed".format(i+1, j+1),end='\r')

def x_sample(i,name):
	a_i = np.sum(X_out[i], axis=0) + np.ones([K])*a_u
	b_i = np.sum(V, axis=0) + np.ones([K])*(b_u)
	U_new[i] = np.random.gamma(a_i, 1./b_i)
def y_sample(j,name):
	a_i = np.sum(X_out[:,j], axis=0) + np.ones([K])*a_v
	b_i = np.sum(U, axis=0) + np.ones([K])*(b_v)
	V_new[j] = np.random.gamma(a_i,1./ b_i)
	

for i in range(num_iterations):
	start_iter = time.time()
	print("Running iteration : {}".format(i + 1))
	print("Sampling from multinomial")
	threads = []
	for key in keys:
		print("Setting up thread: {}".format(key), end='\r')
		thread = Thread(target=sample,args=(key[0],key[1],str(i)))
		threads.append(thread)
	x = 0
	start = time.time()
	while(x < len(threads)):
			threads[x].start()
			x+=1
	for i in range(len(threads)):
		threads[i].join()
	print(time.time() - start)
	print("sampling from gamma for u")
	threads = []
	for i in range(N):
		thread = Thread(target=x_sample,args=(i, str(i)))
		threads.append(thread)
	x = 0
	start = time.time()
	while(x < len(threads)):
		if (threading.active_count() < 32):
			threads[x].start()
			x+=1
	for i in range(len(threads)):
		threads[i].join()
	print(time.time() - start)
	print("sampling from gamma for v")
	threads = []
	for j in range(M):
		thread = Thread(target=y_sample,args= (j, str(j)))
		threads.append(thread)
	x = 0
	start = time.time()
	while(x < len(threads)):
		if (threading.active_count() < 32):
			threads[x].start()
			x+=1
	for i in range(len(threads)):
		threads[i].join()
	print("computing error and seeing results")
	X_new = np.matmul(U, np.transpose(V))
	X_sparse_new = dok_matrix(sparse)
	error = np.mean(np.abs(sparse-X_sparse_new))
	if error_old == None or i < 50:
		sparse_old = X_sparse_new
	else :
		sparse_old = sparse_old * (float(i)/float(i+1)) + (1./i+1)*X_sparse_new
	error_old = np.mean(np.abs(sparse-sparse_old))
	print("Error is : {} in time : {}".format(error, time.time() - start_iter))
	print("Error averaged is : {} in time : {}".format(error_old, time.time() - start_iter))
	U = np.copy(U_new)
	V = np.copy(V_new)
