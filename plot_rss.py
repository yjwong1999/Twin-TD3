import os
import matplotlib.pyplot as plt
import numpy as np

dir_ = 'Secrecy_rate'
files = os.listdir(dir_)
files = sorted(files)
files = files[-1 * 2 * 100 * 100:]

total = []
for i in range(0, 20000, 2):
    try:
        x=np.load(os.path.join('Secrecy_rate',files[i]),allow_pickle=True)
        y=np.load(os.path.join('Secrecy_rate',files[i+1]),allow_pickle=True)
        _ = x + y
        total.append(_)
    except:
        print(i)
        total.append(total[-1])
plt.plot(total)
plt.show()



ave_total = []
for i in range(0,10000, 100):
    x = total[i:i+100]
    _ = sum(x)/len(x)
    ave_total.append(_)

plt.plot(ave_total)
plt.show()
