import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


points = np.array([(1, 1560), (2, 1110), (3, 900), (4, 875), (5, 510)])
# get x and y vectors
x = points[:,0]
y = points[:,1]

# calculate polynomial
z = np.polyfit(x, y, 2)
f = np.poly1d(z)
print f
def integrand(x, a, b, c):
    return (a*x**2) + (b * x) + c
I = quad(integrand, 1, 5, args=(f[2], f[1], f[0]))
print I[0] * .001

# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 50)
y_new = f(x_new)

plt.plot(x,y,'o', x_new, y_new)
plt.xlim([x[0]-1, x[-1] + 1 ])
plt.show()
