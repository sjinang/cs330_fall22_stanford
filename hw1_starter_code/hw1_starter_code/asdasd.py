import numpy as np 

a = np.zeros((10,20))
a = np.ones((3,3,3))
a[:,-1,:]=0
print(a)
