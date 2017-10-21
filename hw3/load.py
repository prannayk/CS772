import numpy as np
import scipy.io as sci
import scipy.stats as stats

def loader(path):
	m = sci.loadmat(path)
	dictionary = {}
	for t in m.items():
		dictionary[t[0]] = t[1]
	return dictionary
