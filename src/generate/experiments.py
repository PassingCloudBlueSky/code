import numpy as np
import matplotlib.pyplot as plt

a=np.arange(10)

b=np.empty((1,len(a)))
print(b.shape)

print(a[2:4])

f=np.arange(1,6,0.01)
idx=np.argmin(np.abs(f-2))
print(f[idx:])