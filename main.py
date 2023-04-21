import numpy as np 
import matplotlib.pyplot as plt

# create a numpy array with some data
x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)

# create a plot using matplotlib
plt.plot(x, y)
plt.title("A Sine Wave")
plt.xlabel("x")
plt.ylabel("y")
plt.show()