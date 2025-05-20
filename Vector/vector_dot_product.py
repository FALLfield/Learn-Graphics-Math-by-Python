import numpy as np
import matplotlib.pyplot as plt

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors using dot product"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # According to the formula of dot product, the cosine of the angle between two vectors is: cos(θ) = (v1·v2)/(|v1|·|v2|)
    #  v1·v2 is the dot product of two vectors, |v1| and |v2| are the magnitudes of two vectors
    # the formula is derived from the geometric definition of dot product: v1·v2 = |v1|·|v2|·cos(θ)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    
    # Use the inverse cosine function (arccos) to calculate the angle from the cosine value
    # np.clip function limits the cosine value to the range [-1,1], to avoid calculation problems due to floating-point errors
    # For example, due to precision issues, cos_angle may be slightly greater than 1 or slightly less than -1, which will cause the arccos function to fail
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def project_vector(v1, v2):
    """Project vector v1 onto vector v2"""
    dot_product = np.dot(v1, v2)
    norm_v2 = np.linalg.norm(v2)
    projection_length = dot_product / norm_v2
    projection = (projection_length / norm_v2) * v2
    return projection

def plot_vectors_and_projection(v1, v2, projection):
    """Plot the vectors and their projection"""
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    # Plot original vectors
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')
    
    # Plot projection
    projection_arrow = plt.quiver(0, 0, projection[0], projection[1], angles='xy', scale_units='xy', scale=1, 
              color='g', label='Vector 1\'s Projection on Vector 2')
    
    # Plot perpendicular component
    perpendicular = v1 - projection
    perpendicular_arrow = plt.quiver(0, 0, perpendicular[0], perpendicular[1], angles='xy', scale_units='xy', scale=1,
              color='purple', label='Perpendicular Component')
    
    # Add line connecting v1 and projection
    plt.plot([0, v1[0]], [0, v1[1]], 'r--', alpha=0.5)
    plt.plot([0, projection[0]], [0, projection[1]], 'g--', alpha=0.7)
    plt.plot([0, perpendicular[0]], [0, perpendicular[1]], ':', color='purple', alpha=0.7)
    
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Vector Projection and Decomposition')
    plt.legend()
    plt.show()

def analyze_direction_similarity(v1, v2):
    """Analyze how similar two vectors are in direction"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    
    print(f"\nDirection Analysis:")
    print(f"Dot product: {dot_product:.2f}")
    print(f"Similarity (cosine): {similarity:.2f}")
    
    if similarity > 0:
        print("Vectors are pointing in similar directions")
        if similarity > 0.9:
            print("Vectors are almost parallel")
    elif similarity < 0:
        print("Vectors are pointing in opposite directions")
        if similarity < -0.9:
            print("Vectors are almost antiparallel")
    else:
        print("Vectors are perpendicular")

# Example vectors
v1 = np.array([3, 2])  # First vector
v2 = np.array([2, 4])  # Second vector

# Calculate angle between vectors
angle = calculate_angle(v1, v2)
print(f"Angle between vectors: {angle:.2f} degrees")

# Calculate projection
projection = project_vector(v1, v2)
print(f"\nProjection of v1 onto v2: {projection}")

# Analyze direction similarity
analyze_direction_similarity(v1, v2)

# Plot the vectors and their projection
plot_vectors_and_projection(v1, v2, projection) 