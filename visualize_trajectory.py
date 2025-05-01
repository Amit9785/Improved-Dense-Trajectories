import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_trajectory(file_path, interactive=True):
    # Enable interactive mode if requested
    if interactive:
        plt.ion()
        
    # Load the trajectory data
    trajectory_data = np.load(file_path)

    # Print information about the loaded data
    print(f"Loaded {len(trajectory_data)} trajectories")
    if len(trajectory_data) > 0:
        print(f"First trajectory has {len(trajectory_data[0])} elements")
        
        # Print detailed information about the first trajectory's structure
        print("\n=== First trajectory complete data structure ===")
        for i, item in enumerate(trajectory_data[0]):
            print(f"Element {i}: {type(item)} - {item}")
        
        # Print specific information about points (index 9) and motion vectors (index 10)
        if len(trajectory_data[0]) > 9:
            points_data = trajectory_data[0][9]
            print("\n=== Points data (index 9) ===")
            print(f"Type: {type(points_data)}")
            print(f"Length: {len(points_data)}")
            print("First 5 coordinate pairs:")
            for i, point in enumerate(points_data[:5]):
                print(f"  Point {i}: {point} (x={point[0]}, y={point[1]})")
        
        if len(trajectory_data[0]) > 10:
            motion_data = trajectory_data[0][10]
            print("\n=== Motion vectors data (index 10) ===")
            print(f"Type: {type(motion_data)}")
            print(f"Length: {len(motion_data)}")
            print("First 5 motion vectors:")
            for i, vector in enumerate(motion_data[:5]):
                print(f"  Vector {i}: {vector} (dx={vector[0]}, dy={vector[1]})")
        
        print("\n=== Relationship between points and motion vectors ===")
        if len(trajectory_data[0]) > 10:
            points_data = trajectory_data[0][9]
            motion_data = trajectory_data[0][10]
            print(f"Number of points: {len(points_data)}")
            print(f"Number of motion vectors: {len(motion_data)}")
            print(f"Expected relationship: motion vectors = points - 1? {len(motion_data) == len(points_data) - 1}")
        
        print("\n=== Continuing with visualization ===")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Process each trajectory
    for i, trajectory in enumerate(trajectory_data):
        # Extract point coordinates (10th element, index 9)
        # The data is a list of [x,y] coordinates
        try:
            # Convert the list of coordinates to numpy array
            try:
                points = np.array(trajectory[9])
                
                if len(points) == 0:
                    print(f"Skipping trajectory {i}: No points found")
                    continue
                    
                # Print debug info about the points
                print(f"Debug: Trajectory {i} points shape: {np.array(points).shape}")
                print(f"Debug: First point: {points[0]}")
                
                # Ensure points are properly shaped as a 2D array of [x,y] coordinates
                if len(points.shape) != 2 or points.shape[1] != 2:
                    print(f"Trajectory {i} points need reshaping from {points.shape}")
                    # Try to fix the shape if possible
                    if len(points.shape) == 1:
                        # If it's a 1D array, it might be flattened [x1,y1,x2,y2,...]
                        if len(points) % 2 == 0:
                            points = points.reshape(-1, 2)
                        else:
                            print(f"Cannot reshape points of length {len(points)}")
                            continue
                    else:
                        print(f"Cannot handle points shape {points.shape}")
                        continue
                
                print(f"Using points with shape {points.shape}")
            except IndexError as e:
                print(f"Error accessing point data in trajectory {i}: {e}")
                continue
            except ValueError as e:
                print(f"Error with point data format in trajectory {i}: {e}")
                continue
            except Exception as e:
                print(f"Error processing point data in trajectory {i}: {e}")
                continue
            
            # Extract motion vectors (11th element, index 10)
            try:
                motion_vectors = np.array(trajectory[10])
                
                if len(motion_vectors) == 0:
                    print(f"Skipping trajectory {i}: No motion vectors found")
                    continue
                    
                # Print debug info about the motion vectors
                print(f"Debug: Trajectory {i} motion vectors shape: {motion_vectors.shape}")
                print(f"Debug: First motion vector: {motion_vectors[0]}")
                
                # Ensure motion vectors are properly shaped
                if len(motion_vectors.shape) != 2 or motion_vectors.shape[1] != 2:
                    print(f"Trajectory {i} motion vectors need reshaping from {motion_vectors.shape}")
                    # Try to fix the shape if possible
                    if len(motion_vectors.shape) == 1:
                        # If it's a 1D array, it might be flattened [dx1,dy1,dx2,dy2,...]
                        if len(motion_vectors) % 2 == 0:
                            motion_vectors = motion_vectors.reshape(-1, 2)
                        else:
                            print(f"Cannot reshape motion vectors of length {len(motion_vectors)}")
                            continue
                    else:
                        print(f"Cannot handle motion vectors shape {motion_vectors.shape}")
                        continue
                
                print(f"Using motion vectors with shape {motion_vectors.shape}")
            except IndexError as e:
                print(f"Error accessing motion vector data in trajectory {i}: {e}")
                continue
            except ValueError as e:
                print(f"Error with motion vector data format in trajectory {i}: {e}")
                continue
            except Exception as e:
                print(f"Error processing motion vector data in trajectory {i}: {e}")
                continue
                        # Print debug info if shapes don't match
                        if len(points) != len(motion_vectors) + 1:
                            print(f"Warning: Trajectory {i} has {len(points)} points but {len(motion_vectors)} vectors")
        
        # Extract x and y coordinates
        try:
            x = points[:, 0]
            y = points[:, 1]
            
            # Extract motion vector components
            dx = motion_vectors[:, 0]
            dy = motion_vectors[:, 1]
        except IndexError as e:
            print(f"Error extracting coordinates from trajectory {i}: {e}")
            print(f"Points shape: {points.shape}, Motion vectors shape: {motion_vectors.shape}")
            continue
        
        # Plot trajectory points
        ax.plot(x, y, 'o-', linewidth=1, markersize=3, alpha=0.7, 
            label=f'Trajectory {i+1}' if i < 5 else "")
        
        # Plot motion vectors (scale them for better visualization)
        scale = 1.0
        # Make sure we don't try to plot more vectors than we have points
        n_vecs = min(len(x)-1, len(dx))
        ax.quiver(x[:n_vecs], y[:n_vecs], dx[:n_vecs]*scale, dy[:n_vecs]*scale, angles='xy', 
                scale_units='xy', scale=1, color='r', alpha=0.5, width=0.003)
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectory Visualization with Motion Vectors')
    
    # Add grid and legend (limit to first 5 trajectories to avoid overcrowding)
    ax.grid(True, linestyle='--', alpha=0.7)
    if trajectory_data.shape[0] > 0:
        ax.legend(loc='upper right', ncol=1, fontsize='small')
    
    # Equal aspect ratio for better visualization
    ax.set_aspect('equal')
    
    # Show the plot
    plt.tight_layout()

    if interactive:
        # In interactive mode, draw the plot but don't block
        plt.draw()
        plt.pause(0.001)  # Small pause to allow the UI to update
        input("Press Enter to close the plot...")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Path to the trajectory data file
    file_path = 'features/person01_boxing_d1_uncomp-trajectory.npy'

    # Visualize the trajectory (set interactive=True for interactive mode)
    visualize_trajectory(file_path, interactive=True)

