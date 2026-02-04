"""
Generate Kepler orbit trajectories and save them to disk.

This script generates Kepler orbit trajectories and saves them in the data_cv folder
for use in training.
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
import torch
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable

# Configuration
seed = 1
np.random.seed(seed)

# Parameters for Kepler orbits
num_points_per_trajectory = 100
dt = 0.2  # Time step for integration


def kepler_orbit(t, state, GM=1.0):
    """
    Kepler orbit equations: d^2r/dt^2 = -GM * r / |r|^3
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    r3 = r**3
    ax = -GM * x / r3
    ay = -GM * y / r3
    return [vx, vy, ax, ay]


def generate_kepler_trajectory(eccentricity=0.5, semi_major_axis=1.0, angle=0.0, GM=1.0, dt=0.2, num_points=100):
    """
    Generate a Kepler orbit trajectory.
    
    Args:
        eccentricity: Eccentricity of the ellipse (0 = circle, <1 = ellipse)
        semi_major_axis: Semi-major axis of the ellipse
        angle: Initial angle of the orbit
        GM: Gravitational parameter (mass of sun * gravitational constant)
        dt: Time step
        num_points: Number of points in the trajectory
    
    Returns:
        positions: array of shape (num_points, 2) with (x, y) positions
        velocities: array of shape (num_points, 2) with (vx, vy) velocities
    """
    # Initial conditions for elliptical orbit
    # At perihelion (closest point to sun)
    r_peri = semi_major_axis * (1 - eccentricity)
    v_peri = np.sqrt(GM * (1 + eccentricity) / (semi_major_axis * (1 - eccentricity)))
    
    # Rotate by angle
    x0 = r_peri * np.cos(angle)
    y0 = r_peri * np.sin(angle)
    vx0 = -v_peri * np.sin(angle)
    vy0 = v_peri * np.cos(angle)
    
    initial_state = [x0, y0, vx0, vy0]
    
    # Integrate orbit
    t_span = (0, num_points * dt)
    t_eval = np.linspace(0, num_points * dt, num_points)
    
    sol = solve_ivp(kepler_orbit, t_span, initial_state, t_eval=t_eval, 
                    args=(GM,), rtol=1e-8, atol=1e-8)
    
    positions = np.column_stack([sol.y[0], sol.y[1]])  # (num_points, 2)
    velocities = np.column_stack([sol.y[2], sol.y[3]])  # (num_points, 2)
    return positions, velocities


def generate_kepler_trajectory_euler(eccentricity=0.5, semi_major_axis=1.0, angle=0.0, GM=1.0, dt=0.2, num_points=100):
    """
    Generate a Kepler orbit trajectory using forward Euler integration.
    This serves as a baseline model with simple explicit integration.
    
    Args:
        eccentricity: Eccentricity of the ellipse (0 = circle, <1 = ellipse)
        semi_major_axis: Semi-major axis of the ellipse
        angle: Initial angle of the orbit
        GM: Gravitational parameter (mass of sun * gravitational constant)
        dt: Time step for forward Euler integration
        num_points: Number of points in the trajectory
    
    Returns:
        positions: array of shape (num_points, 2) with (x, y) positions
        velocities: array of shape (num_points, 2) with (vx, vy) velocities
    """
    # Initial conditions for elliptical orbit
    # At perihelion (closest point to sun)
    r_peri = semi_major_axis * (1 - eccentricity)
    v_peri = np.sqrt(GM * (1 + eccentricity) / (semi_major_axis * (1 - eccentricity)))
    
    # Rotate by angle
    x0 = r_peri * np.cos(angle)
    y0 = r_peri * np.sin(angle)
    vx0 = -v_peri * np.sin(angle)
    vy0 = v_peri * np.cos(angle)
    
    # Initialize arrays to store positions and velocities
    positions = np.zeros((num_points, 2))
    velocities = np.zeros((num_points, 2))
    
    # Initial state: [x, y, vx, vy]
    state = np.array([x0, y0, vx0, vy0], dtype=np.float64)
    
    # Forward Euler integration
    for i in range(num_points):
        # Store current state
        positions[i] = state[:2]
        velocities[i] = state[2:]
        
        # Compute derivatives using kepler_orbit function
        dstate_dt = np.array(kepler_orbit(0, state, GM), dtype=np.float64)
        
        # Forward Euler step: state_new = state_old + dt * dstate_dt
        state = state + dt * dstate_dt
    
    return positions, velocities


def generate_trajectories(num_trajectories=100, chunk_seed_offset=0):
    """
    Generate multiple trajectories with varying parameters.
    
    Args:
        num_trajectories: Number of trajectories to generate
        chunk_seed_offset: Offset for random seed (for chunk generation)
    
    Returns:
        trajectories: array of shape (num_trajectories, num_points, 2)
        orbital_params: list of dicts with orbital parameters for each trajectory
    """
    trajectories = []
    trajectories_euler = []
    orbital_params = []
    # Use different seed for each chunk to ensure variety
    np.random.seed(seed + chunk_seed_offset)
    for i in range(num_trajectories):
        e = np.random.uniform(0.0, 0.8)  # Eccentricity
        a = np.random.uniform(0.5, 2.0)  # Semi-major axis
        angle = np.random.uniform(0, 2 * np.pi)  # Initial angle
        traj, velocities = generate_kepler_trajectory(eccentricity=e, semi_major_axis=a, angle=angle, 
                                         dt=dt, num_points=num_points_per_trajectory)
        traj_euler, velocities_euler = generate_kepler_trajectory_euler(eccentricity=e, semi_major_axis=a, angle=angle, 
                                         dt=dt, num_points=num_points_per_trajectory)
        trajectories.append(traj)
        trajectories_euler.append(traj_euler)
        # Calculate orbital parameters
        b = a * np.sqrt(1 - e**2)  # Semi-minor axis
        c = e * a  # Linear eccentricity
        average_radius = np.sqrt(a * b)  # Average radius
        
        # Calculate Laplace-Runge-Lenz vector
        # Use the first point to compute LRL vector (it's conserved throughout the orbit)
        x, y = traj[0]
        vx, vy = velocities[0]
        r = np.sqrt(x**2 + y**2)
        Lz = x * vy - y * vx  # Angular momentum (z-component)
        
        # LRL vector: A = v × L - GM * (r / |r|)
        # For 2D: A = (vy*Lz - GM*x/r, -vx*Lz - GM*y/r)
        GM = 1.0  # Gravitational parameter
        Ax = vy * Lz - GM * x / r
        Ay = -vx * Lz - GM * y / r
        A_magnitude = np.sqrt(Ax**2 + Ay**2)
        A_angle = np.arctan2(Ay, Ax)  # Direction of LRL vector
        
        orbital_params.append({
            'e': float(e),
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'average_radius': float(average_radius),
            'LRL_x': float(Ax),
            'LRL_y': float(Ay),
            'LRL_magnitude': float(A_magnitude),
            'LRL_angle': float(A_angle),
            'n_x': np.cos(angle),
            'n_y': np.sin(angle)
        })
        
        if i % 10000 == 0 and i > 0:
            print(f"Generated {i} trajectories")
    
    trajectories_euler = np.array(trajectories_euler)
    trajectories = np.array(trajectories)
    euler_error = np.mean(np.sqrt(np.sum((trajectories_euler - trajectories)**2, axis=2)), axis=0)
    return np.array(trajectories), orbital_params, euler_error  # Shape: (num_trajectories, num_points, 2)


def generate_and_save_dataset(num_trajectories=20000, output_dir='data_cv', chunk_size=10000):
    """
    Generate and save Kepler trajectories in chunks.
    
    Args:
        num_trajectories: Total number of trajectories to generate
        output_dir: Output directory for saving trajectories
        chunk_size: Number of trajectories per chunk
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_trajectories:,} Kepler orbit trajectories...")
    print(f"Chunk size: {chunk_size:,} trajectories")
    print(f"Total chunks: {(num_trajectories + chunk_size - 1) // chunk_size}")
    
    # Calculate total memory needed (rough estimate)
    total_memory_gb = (num_trajectories * num_points_per_trajectory * 2 * 4) / (1024**3)  # 4 bytes per float32
    print(f"Estimated dataset size: {total_memory_gb:.2f} GB")
    
    num_chunks = (num_trajectories + chunk_size - 1) // chunk_size
    
    # Initialize range tracking variables
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for chunk_idx in tqdm(range(num_chunks), desc="Generating chunks"):
        # Calculate trajectories for this chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_trajectories)
        chunk_num_trajectories = end_idx - start_idx
        
        # Generate data for this chunk
        trajectories, orbital_params, euler_error = generate_trajectories(chunk_num_trajectories, chunk_seed_offset=chunk_idx)
        
        # Save trajectory chunk to disk
        chunk_filename = os.path.join(output_dir, f'trajectories_chunk_{chunk_idx:06d}.pt')
        chunk_filename_euler = os.path.join(output_dir, f'euler_error_chunk_{chunk_idx:06d}.pt')
        torch.save(torch.from_numpy(trajectories).float(), chunk_filename)
        torch.save(torch.from_numpy(euler_error).float(), chunk_filename_euler)
        
        # Save orbital parameters chunk to disk
        orbital_params_chunk_path = os.path.join(output_dir, f'orbital_params_chunk_{chunk_idx:06d}.py')
        with open(orbital_params_chunk_path, 'w') as f:
            f.write("# Orbital parameters for Kepler trajectories (chunk)\n")
            f.write("# Generated by generate_kepler_cv.py\n")
            f.write("# Format: list of dicts with keys: e, a, b, c, average_radius, LRL_x, LRL_y, LRL_magnitude, LRL_angle, n_x, n_y\n\n")
            f.write("orbital_params = [\n")
            for i, params in enumerate(orbital_params):
                comma = "," if i < len(orbital_params) - 1 else ""
                f.write(f"    {{'e': {params['e']:.10f}, 'a': {params['a']:.10f}, "
                       f"'b': {params['b']:.10f}, 'c': {params['c']:.10f}, "
                       f"'average_radius': {params['average_radius']:.10f}, "
                       f"'LRL_x': {params['LRL_x']:.10f}, 'LRL_y': {params['LRL_y']:.10f}, "
                       f"'LRL_magnitude': {params['LRL_magnitude']:.10f}, "
                       f"'LRL_angle': {params['LRL_angle']:.10f}, "
                       f"'n_x': {params['n_x']:.10f}, 'n_y': {params['n_y']:.10f}}}{comma}\n")
            f.write("]\n")
        
        # Track position ranges for metadata
        x_min = min(x_min, float(trajectories[:,:,0].min()))
        x_max = max(x_max, float(trajectories[:,:,0].max()))
        y_min = min(y_min, float(trajectories[:,:,1].min()))
        y_max = max(y_max, float(trajectories[:,:,1].max()))
        
        # Clean up memory
        del trajectories
        del orbital_params
    
    # Save metadata
    metadata = {
        'num_trajectories': num_trajectories,
        'num_points_per_trajectory': num_points_per_trajectory,
        'chunk_size': chunk_size,
        'num_chunks': num_chunks,
        'dt': dt,
        'seed': seed,
        'x_range': [float(x_min), float(x_max)],
        'y_range': [float(y_min), float(y_max)]
    }
    metadata_path = os.path.join(output_dir, 'metadata.pt')
    torch.save(metadata, metadata_path)
    
    print(f"\nDataset generation complete!")
    print(f"Saved {num_chunks} trajectory chunks to {output_dir}")
    print(f"Saved {num_chunks} orbital parameter chunks to {output_dir}")
    print(f"Position range: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"Metadata saved to {metadata_path}")


def main():
    """Generate and save Kepler trajectories."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Kepler orbit trajectories')
    parser.add_argument('--num_trajectories', type=int, default=20000, 
                       help='Number of trajectories to generate')
    parser.add_argument('--output_dir', type=str, default='data_cv',
                       help='Output directory for saving trajectories')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Number of trajectories per chunk')
    args = parser.parse_args()
    
    generate_and_save_dataset(
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size
    )
    
    print("Done!")


if __name__ == "__main__":
    main()

