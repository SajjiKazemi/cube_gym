import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw3d_target_cube(location: np.array, color= str):
    # Define the vertices of the target cube (8 vertices)
    x, y, z = location[0], location[1], location[2]
    vertices = [
        [x, y, z],
        [x + 1, y, z],
        [x + 1, y + 1, z],
        [x, y + 1, z],
        [x, y, z + 1],
        [x + 1, y, z + 1],
        [x + 1, y + 1, z + 1],
        [x, y + 1, z + 1]
    ]

    # Define the faces of the cube using the vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]], # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]], # Side faces
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    # Create a Poly3DCollection and add it to the plot
    return Poly3DCollection(faces, linewidths=1, facecolor = color,edgecolor=color, alpha=0.2)


def draw3d_agent_sphere(center, radius, num_points=10):
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    return x, y, z