from __future__ import print_function
import numpy as np
import sys
fname = sys.argv[1]

with open("vocab.txt") as f :
	text = f.readlines()
	vocab = map(lambda x : str(x).split("\n")[0], text)

mat = np.load(fname)
for i in range(mat.shape[-1]):
	a = mat[:,i]
	dict_words = {}
	for j in range(2000):
		dict_words[vocab[j]] = a[j]
	sorted_array = sorted(dict_words.iteritems(), key=lambda (k,v) : -v)
	with open("words-{}.txt".format(i), mode="a") as f:
		f.write('\n'.join(map(lambda (k,v) : str(k),sorted_array[:50])))
	print("Completed for iteration : {}".format(i), end='\r')
