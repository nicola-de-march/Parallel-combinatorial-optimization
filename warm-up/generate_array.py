import numpy as np

# Define the sizes of the arrays
sizes = [100, 500, 1000, 5000, 10000, 100000, 500000, 1000000]

# Generate and save the arrays
for size in sizes:
  array = np.random.randint(0, 1000, size)
  filename = f'arrays/array_{size}.txt'
  np.savetxt(filename, array, fmt='%d')