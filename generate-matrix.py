import random
import numpy as np

"""
Utility file to generate and save matrices to be used for testing
"""

n100 = np.random.randint(1, 20, (100,100))
n1000 = np.random.randint(1, 20, (1000,1000))
n5000 = np.random.randint(1, 20, (5000,5000))
n10000 = np.random.randint(1, 20, (10000,10000))

np.savez_compressed("n-100", n100)
np.savez_compressed("n-1000", n1000)
np.savez_compressed("n-5000", n5000)
np.savez_compressed("n-10000", n10000)
