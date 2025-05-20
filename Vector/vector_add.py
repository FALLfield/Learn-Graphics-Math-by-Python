import numpy as np
import matplotlib.pyplot as plt

# Create two vectors
v1 = np.array([2, 3])  # The first vector
v2 = np.array([3, 1])  # The second vector
v_sum = v1 + v2        # Add the two vectors

# Create a graph
plt.figure(figsize=(8, 6))
plt.grid(True)

# Draw vectors
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')
#Move the vectors' origins to the ends of each vectors
plt.quiver(3, 1, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='grey', label='Transformed Vector 1')
plt.quiver(2, 3, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='grey', label='Transformed Vector 2')
plt.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='g', label='Vector Sum')

# Set the range of the coordinate axis
plt.xlim(0, 6)
plt.ylim(0, 6)

# Add labels and legends
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Vector Addition Visualization')
plt.legend()

# Display the graph
plt.show()

