import numpy as np
import matplotlib.pyplot as plt
with open("loss.txt") as f:
	text = f.readlines()
text = map(lambda x: np.array(str(x).split("\n")[0].split(","), dtype=np.float32), text)
data = np.array(text)
for i in range(data.shape[0]):
	plt.scatter(i,np.mean(data[:i,1]))
plt.show()
