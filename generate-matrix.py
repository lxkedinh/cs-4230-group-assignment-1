import random
import numpy as np

"""
Utility file to generate and save matrices to be used for testing
"""

n10a = np.random.randint(1, 20, (10,10))
n10b = np.random.randint(1, 20, (10,10))
n100a = np.random.randint(1, 20, (100,100))
n100b = np.random.randint(1, 20, (100,100))
n1000a = np.random.randint(1, 20, (1000,1000))
n1000b = np.random.randint(1, 20, (1000,1000))
n5000a = np.random.randint(1, 20, (5000,5000))
n5000b = np.random.randint(1, 20, (5000,5000))
n10000a = np.random.randint(1, 20, (10000,10000))
n10000b = np.random.randint(1, 20, (10000,10000))

np.savez_compressed("n-10a", n10a)
np.savez_compressed("n-10b", n10b)
np.savez_compressed("n-100a", n100a)
np.savez_compressed("n-100b", n100b)
np.savez_compressed("n-1000a", n1000a)
np.savez_compressed("n-1000b", n1000b)
np.savez_compressed("n-5000a", n5000a)
np.savez_compressed("n-5000b", n5000b)
np.savez_compressed("n-10000a", n10000a)
np.savez_compressed("n-10000b", n10000b)
