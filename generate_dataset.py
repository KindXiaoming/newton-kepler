"""
Generate and save a large dataset of sine wave trajectories.

This script generates 2 * 10^9 trajectories and saves them to disk for reuse.
"""

import torch
import numpy as np
import os
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable


def generate_sine_wave_data(
    num_trajectories=100,
    num_points_per_trajectory=100,
    omega_s=0.5,
    omega_l=2,
    A_s=0.5,
    A_l=1.0,
    phi_s=0,
    phi_l=2 * np.pi,
    t_max=20,
    seed=1
):
    """
    Generate sine wave trajectories with random parameters.
    
    Args:
        num_trajectories: Number of trajectories to generate
        num_points_per_trajectory: Number of points per trajectory
        omega_s, omega_l: Frequency range [omega_s, omega_l]
        A_s, A_l: Amplitude range [A_s, A_l]
        phi_s, phi_l: Phase range [phi_s, phi_l]
        t_max: Maximum time value
        seed: Random seed
        
    Returns:
        ts: Time points (num_points_per_trajectory,)
        xs: Trajectories (num_trajectories, num_points_per_trajectory)
        omegas: Frequencies used (num_trajectories,)
        As: Amplitudes used (num_trajectories,)
        phis: Phases used (num_trajectories,)
    """
    np.random.seed(seed)
    
    ts = np.linspace(0, t_max, num_points_per_trajectory)
    
    # Draw random parameters
    omegas = np.random.uniform(omega_s, omega_l, size=num_trajectories)
    As = np.random.uniform(A_s, A_l, size=num_trajectories)
    phis = np.random.uniform(phi_s, phi_l, size=num_trajectories)
    
    # Generate sine waves
    xs = As[:, None] * np.sin(omegas[:, None] * ts[None, :] + phis[:, None])
    
    return ts, xs, omegas, As, phis


def bin_trajectories(xs, vocab_size=7000):
    """
    Bin continuous trajectories into discrete tokens.
    
    Args:
        xs: Continuous trajectories (num_trajectories, num_points_per_trajectory)
        vocab_size: Number of bins
        
    Returns:
        xs_binned: Binned trajectories as integers (num_trajectories, num_points_per_trajectory)
        coordinates: Bin center coordinates (vocab_size,)
        inputs_id: Torch tensor of binned trajectories
    """
    # Map x to vocab_size bins
    h = 2 / vocab_size
    xs_binned = np.floor((xs + 1) / h).astype(int)
    coordinates = np.linspace(-1, 1, vocab_size)
    inputs_id = torch.from_numpy(xs_binned).long()
    
    return xs_binned, coordinates, inputs_id


def generate_and_save_dataset(
    num_trajectories=2 * 10**8,
    num_points_per_trajectory=100,
    vocab_size=7000,
    omega_s=0.5,
    omega_l=2,
    A_s=0.5,
    A_l=1.0,
    phi_s=0,
    phi_l=2 * np.pi,
    t_max=20,
    seed=1,
    output_dir='./data',
    chunk_size=10**6  # Generate in chunks to manage memory
):
    """
    Generate a large dataset and save it in chunks.
    
    Args:
        num_trajectories: Total number of trajectories to generate
        num_points_per_trajectory: Number of points per trajectory
        vocab_size: Vocabulary size for binning
        output_dir: Directory to save the dataset
        chunk_size: Number of trajectories to generate per chunk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_trajectories:,} trajectories...")
    print(f"Chunk size: {chunk_size:,} trajectories")
    print(f"Total chunks: {(num_trajectories + chunk_size - 1) // chunk_size}")
    
    # Calculate total memory needed (rough estimate)
    total_memory_gb = (num_trajectories * num_points_per_trajectory * 8) / (1024**3)  # 8 bytes per long int
    print(f"Estimated dataset size: {total_memory_gb:.2f} GB")
    
    all_inputs_id = []
    num_chunks = (num_trajectories + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(num_chunks), desc="Generating chunks"):
        # Calculate trajectories for this chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_trajectories)
        chunk_num_trajectories = end_idx - start_idx
        
        # Generate data for this chunk
        ts, xs, omegas, As, phis = generate_sine_wave_data(
            num_trajectories=chunk_num_trajectories,
            num_points_per_trajectory=num_points_per_trajectory,
            omega_s=omega_s,
            omega_l=omega_l,
            A_s=A_s,
            A_l=A_l,
            phi_s=phi_s,
            phi_l=phi_l,
            t_max=t_max,
            seed=seed + chunk_idx  # Different seed per chunk for variety
        )
        
        # Bin trajectories
        xs_binned, coordinates, inputs_id = bin_trajectories(xs, vocab_size=vocab_size)
        
        # Save chunk to disk
        chunk_filename = os.path.join(output_dir, f'inputs_id_chunk_{chunk_idx:06d}.pt')
        torch.save(inputs_id, chunk_filename)
        
        # Optionally save metadata (before deleting coordinates)
        if chunk_idx == 0:
            metadata = {
                'num_trajectories': num_trajectories,
                'num_points_per_trajectory': num_points_per_trajectory,
                'vocab_size': vocab_size,
                'chunk_size': chunk_size,
                'num_chunks': num_chunks,
                'omega_s': omega_s,
                'omega_l': omega_l,
                'A_s': A_s,
                'A_l': A_l,
                'phi_s': phi_s,
                'phi_l': phi_l,
                't_max': t_max,
                'seed': seed,
                'coordinates': coordinates
            }
            torch.save(metadata, os.path.join(output_dir, 'metadata.pt'))
        
        # Clean up memory (after saving metadata)
        del ts, xs, omegas, As, phis, xs_binned, coordinates, inputs_id
    
    print(f"\nDataset generation complete!")
    print(f"Saved {num_chunks} chunks to {output_dir}")
    print(f"Metadata saved to {os.path.join(output_dir, 'metadata.pt')}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate and save sine wave dataset')
    parser.add_argument('--num_trajectories', type=int, default=2 * 10**8,
                        help='Total number of trajectories to generate')
    parser.add_argument('--num_points_per_trajectory', type=int, default=100,
                        help='Number of points per trajectory')
    parser.add_argument('--vocab_size', type=int, default=7000,
                        help='Vocabulary size for binning')
    parser.add_argument('--chunk_size', type=int, default=10**6,
                        help='Number of trajectories per chunk')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_and_save_dataset(
        num_trajectories=args.num_trajectories,
        num_points_per_trajectory=args.num_points_per_trajectory,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        seed=args.seed
    )

