
import numpy as np
x1 = np.array([10, 20, 30], float)
print (x1.shape) 
print (x1)

x2 = x1[:, np.newaxis]
print ("shape of x2 is ", x2.shape)
print(x2) 

x3 = x1[np.newaxis, :]
print ( x3.shape)
print (x3)


