import matplotlib.pyplot as plt
import numpy as np

f_r= np.zeros(11)
r= np.arange(0,11,1)

N_max = 10;
P_max = 10;
f_r = 1.9**r*np.exp(-1.9)/np.math.factorial(r)
plt.plot(r, f_r)

