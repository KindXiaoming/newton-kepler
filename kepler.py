"""
Kepler orbit trajectory prediction using discrete tokenized GPT model.

This module loads continuous Kepler orbit trajectories, tokenizes them into bins,
and trains a GPT-based model to predict future positions using cross-entropy loss.
It includes MSE evaluation for comparison with continuous models.
"""

import torch
import numpy as np
import os
import gc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_2d import GPT2DConfig, GPT2D

# Configuration
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

num_points_per_trajectory = 100

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_gpu_memory_stats():
    """Get GPU memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}


def print_gpu_memory_stats(label=""):
    """Print GPU memory usage statistics."""
    stats = get_gpu_memory_stats()
    if torch.cuda.is_available():
        print(f"{label}GPU Memory - Allocated: {stats['allocated_gb']:.3f} GB, "
              f"Reserved: {stats['reserved_gb']:.3f} GB, "
              f"Max Allocated: {stats['max_allocated_gb']:.3f} GB")
    return stats


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_trajectories(data_dir='data_cv', num_trajectories_needed=None):
    """
    Load continuous trajectories from the data_cv folder.
    Supports both chunked format (new) and single file format (old, for backward compatibility).
    
    Args:
        data_dir: Directory containing the saved trajectories
        num_trajectories_needed: Number of trajectories to load (None = load all available)
    
    Returns:
        trajectories: array of shape (num_trajectories, num_points, 2)
    """
    # Check for chunked format first
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        chunk_size = metadata.get('chunk_size', 10000)
        num_chunks = metadata.get('num_chunks', 0)
        total_available = metadata.get('num_trajectories', 0)
        
        # Determine how many trajectories to load
        if num_trajectories_needed is None:
            num_trajectories_needed = total_available
        else:
            num_trajectories_needed = min(num_trajectories_needed, total_available)
        
        print(f"Loading {num_trajectories_needed:,} trajectories from {num_chunks} chunks...")
        
        # Calculate which chunks we need
        num_chunks_needed = (num_trajectories_needed + chunk_size - 1) // chunk_size
        num_chunks_needed = min(num_chunks_needed, num_chunks)
        
        # Load chunks
        chunks = []
        trajectories_loaded = 0
        
        for chunk_idx in range(num_chunks_needed):
            chunk_filename = os.path.join(data_dir, f'trajectories_chunk_{chunk_idx:06d}.pt')
            if os.path.exists(chunk_filename):
                chunk_data = torch.load(chunk_filename, weights_only=False)
                if isinstance(chunk_data, torch.Tensor):
                    chunk_data = chunk_data.numpy()
                
                remaining_needed = num_trajectories_needed - trajectories_loaded
                if chunk_data.shape[0] <= remaining_needed:
                    chunks.append(chunk_data)
                    trajectories_loaded += chunk_data.shape[0]
                else:
                    # Only take what we need
                    chunks.append(chunk_data[:remaining_needed])
                    trajectories_loaded += remaining_needed
                    break
            else:
                print(f"Warning: Chunk {chunk_idx} not found. Stopping.")
                break
        
        # Concatenate chunks
        if chunks:
            trajectories = np.concatenate(chunks, axis=0)
            # Ensure we have exactly the number needed
            if trajectories.shape[0] > num_trajectories_needed:
                trajectories = trajectories[:num_trajectories_needed]
            print(f"Loaded {trajectories.shape[0]:,} trajectories from chunks")
            return trajectories
        else:
            raise FileNotFoundError(f"No dataset chunks found in {data_dir}")
    
    # Fallback to old format (single file) for backward compatibility
    pt_path = os.path.join(data_dir, 'trajectories.pt')
    npy_path = os.path.join(data_dir, 'trajectories.npy')
    
    if os.path.exists(pt_path):
        trajectories = torch.load(pt_path, weights_only=False)
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.numpy()
        
        # Limit to num_trajectories_needed if specified
        if num_trajectories_needed is not None and trajectories.shape[0] > num_trajectories_needed:
            trajectories = trajectories[:num_trajectories_needed]
        
        print(f"Loaded {trajectories.shape[0]:,} trajectories from {pt_path}")
        return trajectories
    elif os.path.exists(npy_path):
        trajectories = np.load(npy_path)
        
        # Limit to num_trajectories_needed if specified
        if num_trajectories_needed is not None and trajectories.shape[0] > num_trajectories_needed:
            trajectories = trajectories[:num_trajectories_needed]
        
        print(f"Loaded {trajectories.shape[0]:,} trajectories from {npy_path}")
        return trajectories
    else:
        raise FileNotFoundError(
            f"Trajectories not found in {data_dir}. "
            f"Please run generate_kepler_cv.py first to generate the dataset."
        )


def compute_bins_and_tokenize(trajectories, vocab_size_x, vocab_size_y):
    """
    Compute bin ranges and tokenize 2D trajectories.
    Each position has two tokens: [x_token, y_token]
    x tokens use vocabulary [0, vocab_size_x-1]
    y tokens use vocabulary [0, vocab_size_y-1] (no offset needed)
    
    Args:
        trajectories: array of shape (num_trajectories, num_points, 2) with (x, y) coordinates
        vocab_size_x: Number of bins for x coordinate
        vocab_size_y: Number of bins for y coordinate
    
    Returns:
        tokenized_trajectories: tensor of shape (num_trajectories, num_points, 2) with [x_token, y_token] at each position
        x_bin_centers: array of shape (vocab_size_x,) with bin centers for x
        y_bin_centers: array of shape (vocab_size_y,) with bin centers for y
        x_range: tuple (x_min, x_max)
        y_range: tuple (y_min, y_max)
    """
    # Compute ranges from all trajectories
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    
    print(f"X range: [{x_min:.6f}, {x_max:.6f}]")
    print(f"Y range: [{y_min:.6f}, {y_max:.6f}]")
    
    # Create uniform bins
    x_bins = np.linspace(x_min, x_max, vocab_size_x + 1)
    y_bins = np.linspace(y_min, y_max, vocab_size_y + 1)
    
    # Compute bin centers
    x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
    
    # Tokenize: assign each coordinate to a bin
    x_coords = trajectories[:, :, 0]  # (num_trajectories, num_points)
    y_coords = trajectories[:, :, 1]  # (num_trajectories, num_points)
    
    # Find bin indices (using searchsorted for efficiency)
    x_tokens = np.searchsorted(x_bins[1:], x_coords, side='right')
    y_tokens = np.searchsorted(y_bins[1:], y_coords, side='right')
    
    # Clamp to valid range [0, vocab_size-1]
    x_tokens = np.clip(x_tokens, 0, vocab_size_x - 1)
    y_tokens = np.clip(y_tokens, 0, vocab_size_y - 1)
    
    # Stack x and y tokens: (num_trajectories, num_points, 2)
    # Each position has [x_token, y_token]
    tokenized = np.stack([x_tokens, y_tokens], axis=-1)  # (num_trajectories, num_points, 2)
    
    # Use optimal integer dtype to save memory
    # int16 can hold values up to 32767, int32 can hold up to 2.1 billion
    max_vocab = max(vocab_size_x, vocab_size_y)
    if max_vocab <= 32767:
        tokenized = tokenized.astype(np.int16)  # 2 bytes per token instead of 8 (int64)
    elif max_vocab <= 2147483647:
        tokenized = tokenized.astype(np.int32)  # 4 bytes per token instead of 8 (int64)
    # else: keep as int64 (default)
    
    # Return as numpy array to save memory - convert to torch tensor only when needed
    return tokenized, x_bin_centers, y_bin_centers, (x_min, x_max), (y_min, y_max)


def tokens_to_continuous(tokenized, x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y):
    """
    Convert tokenized trajectories back to continuous coordinates.
    Each position has two tokens: [x_token, y_token]
    x tokens are in range [0, vocab_size_x-1]
    y tokens are in range [0, vocab_size_y-1]
    
    Since x and y are treated independently, both vocab_size_x and vocab_size_y are required
    for symmetry. This function accepts both parameters to maintain consistency with the
    independent tokenization scheme.
    
    Args:
        tokenized: tensor/array of shape (batch, num_points, 2) with [x_token, y_token] at each position
        x_bin_centers: array of shape (vocab_size_x,) with bin centers for x
        y_bin_centers: array of shape (vocab_size_y,) with bin centers for y
        vocab_size_x: Size of x vocabulary (required for symmetry with vocab_size_y)
        vocab_size_y: Size of y vocabulary (required for symmetry with vocab_size_x)
    
    Returns:
        continuous: array of shape (batch, num_points, 2) with (x, y) coordinates
    """
    if isinstance(tokenized, torch.Tensor):
        tokenized = tokenized.cpu().numpy()
    
    # Ensure tokenized is 3D: (batch, num_points, 2)
    if tokenized.ndim == 2:
        # If shape is (num_points, 2), add batch dimension
        if tokenized.shape[1] == 2:
            tokenized = tokenized.reshape(1, -1, 2)
        else:
            # If shape is (batch, num_points*2), reshape to (batch, num_points, 2)
            tokenized = tokenized.reshape(tokenized.shape[0], -1, 2)
    
    batch_size, num_points, _ = tokenized.shape
    
    # Extract x and y tokens
    x_tokens = tokenized[:, :, 0]  # (batch, num_points)
    y_tokens = tokenized[:, :, 1]  # (batch, num_points)
    
    # Clamp to valid ranges to prevent index errors
    x_tokens = np.clip(x_tokens, 0, vocab_size_x - 1)
    y_tokens = np.clip(y_tokens, 0, vocab_size_y - 1)
    
    # Convert tokens to continuous values using bin centers
    x_continuous = x_bin_centers[x_tokens]  # (batch, num_points)
    y_continuous = y_bin_centers[y_tokens]  # (batch, num_points)
    
    # Stack into (batch, num_points, 2)
    continuous = np.stack([x_continuous, y_continuous], axis=-1)
    
    return continuous


def compute_mse_from_tokens(model, tokenized_inputs, true_continuous, x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y, device, mode='mse', batch_size=128):
    """
    Compute MSE by predicting tokens and converting to continuous values.
    Uses batching to reduce memory usage.
    
    Since x and y are treated independently, both vocab_size_x and vocab_size_y are required
    for symmetry. This function accepts both parameters to maintain consistency with the
    independent tokenization scheme.
    
    Args:
        model: GPT2D model
        tokenized_inputs: tensor/array of shape (batch, num_points, 2) with input tokens (should be on CPU)
        true_continuous: array of shape (batch, num_points, 2) with true continuous coordinates (should be on CPU)
        x_bin_centers: array of shape (vocab_size_x,) with bin centers for x
        y_bin_centers: array of shape (vocab_size_y,) with bin centers for y
        vocab_size_x: Size of x vocabulary (required for symmetry with vocab_size_y)
        vocab_size_y: Size of y vocabulary (required for symmetry with vocab_size_x)
        device: device to run model on
        mode: 'mse' or 'distance'
        batch_size: Batch size for processing (to reduce memory usage)
    
    Returns:
        mse: Mean squared error
    """
    model.eval()
    
    # Ensure tokenized_inputs is numpy array (convert from tensor if needed)
    if isinstance(tokenized_inputs, torch.Tensor):
        tokenized_inputs = tokenized_inputs.cpu().numpy()
    # tokenized_inputs is now guaranteed to be a numpy array
    
    # Convert true_continuous to numpy if it's a tensor
    if isinstance(true_continuous, torch.Tensor):
        true_continuous = true_continuous.cpu().numpy()
    
    # Process in batches to reduce memory usage
    num_samples = tokenized_inputs.shape[0]
    all_errors = []
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            # Extract batch from numpy array, convert to torch tensor and move to GPU only for this batch
            batch_inputs = torch.from_numpy(tokenized_inputs[batch_start:batch_end]).long().to(device)
            batch_true = true_continuous[batch_start:batch_end]
            
            # Get predictions (logits)
            (logits_x, logits_y), _ = model.forward(batch_inputs, targets=None)
            # logits_x shape: (batch, num_points, vocab_size_x)
            # logits_y shape: (batch, num_points, vocab_size_y)
            
            # Get predicted tokens (argmax, zero temperature)
            predicted_x_tokens = torch.argmax(logits_x, dim=-1)  # (batch, num_points)
            predicted_y_tokens = torch.argmax(logits_y, dim=-1)  # (batch, num_points)
            
            # Stack x and y tokens together: (batch, num_points, 2)
            predicted_tokens = torch.stack([predicted_x_tokens, predicted_y_tokens], dim=-1)
            
            # Convert predicted tokens to continuous values
            predicted_continuous = tokens_to_continuous(
                predicted_tokens.cpu().numpy(), x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y
            )
            
            # Compute errors for this batch
            if mode == 'mse':
                if batch_true.shape[1] == predicted_continuous.shape[1]:
                    batch_errors = (predicted_continuous - batch_true) ** 2
                else:
                    # Handle sequence length mismatch
                    min_len = min(predicted_continuous.shape[1], batch_true.shape[1])
                    batch_errors = (predicted_continuous[:, :min_len, :] - batch_true[:, :min_len, :]) ** 2
                all_errors.append(batch_errors.flatten())
            elif mode == 'distance':
                batch_errors = np.linalg.norm(predicted_continuous - batch_true, axis=-1)
                all_errors.append(batch_errors.flatten())
            
            # Clear intermediate tensors immediately
            del logits_x, logits_y, predicted_x_tokens, predicted_y_tokens, predicted_tokens, predicted_continuous
            del batch_inputs
            # Clear GPU cache more frequently for better memory management
            if batch_start % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
    
    # Compute overall MSE
    if mode == 'mse':
        mse = np.mean(np.concatenate(all_errors))
    elif mode == 'distance':
        mse = np.mean(np.concatenate(all_errors))
    
    return mse


def setup_model(vocab_size_x, vocab_size_y, n_layer=2, n_embd=32, device=None):
    """
    Setup and initialize the GPT2D model for 2D tokenized inputs.
    Each position has two tokens: [x_token, y_token]
    x tokens: [0, vocab_size_x-1]
    y tokens: [0, vocab_size_y-1]
    
    Args:
        vocab_size_x: Vocabulary size for x coordinate
        vocab_size_y: Vocabulary size for y coordinate
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        device: Device to place model on
    
    Returns:
        model: GPT2D model instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = GPT2DConfig(
        block_size=num_points_per_trajectory,
        vocab_size_x=vocab_size_x,
        vocab_size_y=vocab_size_y,
        n_layer=n_layer,
        n_head=1,
        n_embd=n_embd,
        bias=True
    )
    
    model = GPT2D(config)
    model = model.to(device)
    return model


def setup_activation_hooks(model):
    """
    Set up hooks to collect intermediate activations from the GPT2D model.
    
    For each transformer block, captures:
    1. Attention output before/after merging into residual stream
    2. MLP output before/after merging into residual stream
    3. MLP hidden activation after GELU
    
    Returns:
        hooks: list of registered hooks
        activation_dict: dictionary to store activations
    """
    activation_dict = {}
    hooks = []
    n_layer = model.config.n_layer
    
    # Helper function to extract tensor from input/output
    def extract_tensor(x):
        if isinstance(x, tuple):
            return x[0].detach()
        return x.detach()
    
    # Hook for attention output BEFORE merging into residual
    def make_attn_output_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_attn_output'] = extract_tensor(output)
        return hook
    
    # Hook for residual stream AFTER attention merge (input to MLP)
    def make_after_attn_merge_hook(block_idx):
        def hook(module, input):
            # MLP receives the residual stream after attention merge
            activation_dict[f'block_{block_idx}_after_attn_merge'] = extract_tensor(input)
        return hook
    
    # Hook for MLP output BEFORE merging into residual
    def make_mlp_output_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_mlp_output'] = extract_tensor(output)
        return hook
    
    # Hook for residual stream AFTER MLP merge (output of block)
    def make_after_mlp_merge_hook(block_idx):
        def hook(module, input, output):
            # Block output is the residual stream after MLP merge
            activation_dict[f'block_{block_idx}_after_mlp_merge'] = extract_tensor(output)
        return hook
    
    # Hook for MLP hidden activation after GELU
    def make_mlp_hidden_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_mlp_hidden'] = extract_tensor(output)
        return hook
    
    # Register hooks for each transformer block
    for block_idx in range(n_layer):
        block = model.transformer.h[block_idx]
        
        # 1. Attention output before merge
        hook = block.attn.register_forward_hook(make_attn_output_hook(block_idx))
        hooks.append(hook)
        
        # 2. Residual after attention merge (input to MLP)
        hook = block.mlp.register_forward_pre_hook(make_after_attn_merge_hook(block_idx))
        hooks.append(hook)
        
        # 3. MLP output before merge
        hook = block.mlp.register_forward_hook(make_mlp_output_hook(block_idx))
        hooks.append(hook)
        
        # 4. Residual after MLP merge (block output)
        hook = block.register_forward_hook(make_after_mlp_merge_hook(block_idx))
        hooks.append(hook)
        
        # 5. MLP hidden activation after GELU
        hook = block.mlp.gelu.register_forward_hook(make_mlp_hidden_hook(block_idx))
        hooks.append(hook)
    
    # Also register hooks for token embeddings and final layer norm
    def make_token_emb_hook():
        def hook(module, input, output):
            activation_dict['token_emb'] = extract_tensor(output)
        return hook
    
    def make_after_pos_emb_hook():
        def hook(module, input, output):
            activation_dict['after_pos_emb'] = extract_tensor(output)
        return hook
    
    def make_after_ln_f_hook():
        def hook(module, input, output):
            activation_dict['after_ln_f'] = extract_tensor(output)
        return hook
    
    # Hook for combined token embeddings (tok_emb_x + tok_emb_y)
    # We'll hook the dropout layer that receives tok_emb + pos_emb
    # Use forward_hook (not pre_hook) to capture the output
    hook = model.transformer.drop.register_forward_hook(make_after_pos_emb_hook())
    hooks.append(hook)
    
    hook = model.transformer.ln_f.register_forward_hook(make_after_ln_f_hook())
    hooks.append(hook)
    
    print(f"Hooks registered for {n_layer} transformer blocks:")
    print(f"  For each block: attn_output, after_attn_merge, mlp_output, after_mlp_merge, mlp_hidden")
    print(f"  Additional: after_pos_emb, after_ln_f")
    print(f"  Total hooks: {len(hooks)}")
    
    return hooks, activation_dict


def compute_gravitational_force(positions):
    """
    Compute gravitational force and related quantities from positions.
    
    Args:
        positions: array of shape (batch, time, 2) with (x, y) positions
    
    Returns:
        gravitational_force: dictionary with force components and related quantities
    """
    positions = np.asarray(positions)
    r = np.sqrt(positions[:,:,0]**2 + positions[:,:,1]**2)
    r3 = r**3
    # Avoid division by zero
    r3 = np.where(r3 < 1e-10, 1e-10, r3)
    Fx = -positions[:,:,0] / r3
    Fy = -positions[:,:,1] / r3
    F_magnitude = np.sqrt(Fx**2 + Fy**2)
    
    # Store gravitational force information
    gravitational_force = {
        'Fx': Fx.flatten(),
        'Fy': Fy.flatten(),
        'F_magnitude': F_magnitude.flatten(),
        'F_direction_x': (Fx / np.where(F_magnitude < 1e-10, 1e-10, F_magnitude)).flatten(),
        'F_direction_y': (Fy / np.where(F_magnitude < 1e-10, 1e-10, F_magnitude)).flatten(),
        # Distance-related quantities
        'r': r.flatten(),                    # r = sqrt(x^2 + y^2)
        'inv_r': (1.0 / np.where(r < 1e-10, 1e-10, r)).flatten(),  # 1/r
        'r_squared': (r**2).flatten(),        # r^2
        'inv_r_squared': (1.0 / np.where(r**2 < 1e-10, 1e-10, r**2)).flatten(),  # 1/r^2
        'inv_r_cubed': (1.0 / np.where(r**3 < 1e-10, 1e-10, r**3)).flatten(),  # 1/r^3
        # Position coordinates
        'x': positions[:,:,0].flatten(),      # x coordinate
        'y': positions[:,:,1].flatten(),      # y coordinate
    }
    
    return gravitational_force


def collect_activations(model, tokenized_inputs, x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y, activation_dict, device):
    """
    Run forward pass to collect activations.
    
    Args:
        model: GPT2D model
        tokenized_inputs: tensor/array of shape (batch, num_points, 2) with tokenized inputs (should be on CPU)
        x_bin_centers: array of bin centers for x coordinate
        y_bin_centers: array of bin centers for y coordinate
        vocab_size_x: Size of x vocabulary
        vocab_size_y: Size of y vocabulary
        activation_dict: dictionary to store activations (populated by hooks)
        device: device to run model on
    
    Returns:
        predictions: model predictions (logits)
        gravitational_force: computed gravitational force quantities
    """
    model.eval()
    
    # Ensure tokenized_inputs is numpy array, convert to torch tensor and move to GPU only for forward pass
    if isinstance(tokenized_inputs, torch.Tensor):
        # If already a tensor, ensure it's on CPU first, then convert to numpy and back
        tokenized_inputs_np = tokenized_inputs.cpu().numpy()
    else:
        # If numpy array, use directly
        tokenized_inputs_np = tokenized_inputs
    
    # Convert to torch tensor and move to GPU only for the forward pass
    tokenized_inputs_gpu = torch.from_numpy(tokenized_inputs_np).long().to(device)
    
    with torch.no_grad():
        (logits_x, logits_y), _ = model.forward(tokenized_inputs_gpu, targets=None)
        
        # Convert tokenized inputs to continuous positions for gravitational force computation
        # Use numpy array to avoid GPU memory usage
        # tokenized_inputs shape: (batch, num_points, 2)
        positions = tokens_to_continuous(
            tokenized_inputs_np, x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y
        )  # (batch, num_points, 2)
        
        # Clear GPU tensor immediately
        del tokenized_inputs_gpu
        torch.cuda.empty_cache()
        
        # Compute gravitational force for comparison
        gravitational_force = compute_gravitational_force(positions)
    
    print("Activations collected:")
    for name in activation_dict.keys():
        print(f"  {name}: {activation_dict[name].shape}")
    
    return (logits_x, logits_y), gravitational_force


def run_linear_probes(activation_dict, gravitational_force):
    """
    Run linear probes to test if intermediate representations contain
    linear directions corresponding to gravitational force.
    
    Returns:
        probe_results: dictionary mapping layer names to probe results
    """
    probe_results = {}
    
    for layer_name, activations in activation_dict.items():
        # Flatten activations: (batch, time, features) -> (batch*time, features)
        # Move to CPU if on GPU before converting to numpy
        act_flat = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
        
        # Probe for different gravitational force components
        probes = {}
        
        # Define probe targets
        probe_targets = [
            'F_magnitude', 'F_direction_x', 'F_direction_y', 'Fx', 'Fy',
            'r', 'inv_r', 'r_squared', 'inv_r_squared', 'inv_r_cubed', 'x', 'y'
        ]
        
        for probe_name in probe_targets:
            probe = LinearRegression()
            probe.fit(act_flat, gravitational_force[probe_name])
            pred = probe.predict(act_flat)
            r2 = r2_score(gravitational_force[probe_name], pred)
            probes[probe_name] = {'r2': r2}
        
        probe_results[layer_name] = probes
    
    # Print results
    print("\nLinear Probe Results (R² scores):")
    print("=" * 80)
    for layer_name, probes in probe_results.items():
        print(f"\n{layer_name}:")
        for probe_name, result in probes.items():
            print(f"  {probe_name:20s}: R² = {result['r2']:.4f}")
    
    return probe_results


def generate_trajectory_and_compute_error(model, tokenized_inputs, true_continuous, x_bin_centers, y_bin_centers, 
                                         vocab_size_x, vocab_size_y, device, conditioning_length=50):
    """
    Generate trajectory continuation using the model for all trajectories.
    
    Args:
        model: trained model
        tokenized_inputs: tokenized input trajectories (for conditioning) - should be on CPU
        true_continuous: full original continuous trajectories (for comparison) - should be on CPU
        x_bin_centers: array of bin centers for x coordinate
        y_bin_centers: array of bin centers for y coordinate
        vocab_size_x: Size of x vocabulary
        vocab_size_y: Size of y vocabulary
        device: device to run model on
        conditioning_length: number of initial steps to use as conditioning
    
    Returns:
        error_stats: dictionary with aggregated error statistics across all trajectories
    """
    model.eval()
    
    # Ensure tokenized_inputs is numpy array
    if isinstance(tokenized_inputs, torch.Tensor):
        tokenized_inputs = tokenized_inputs.cpu().numpy()
    # tokenized_inputs is now guaranteed to be a numpy array
    
    num_trajectories = tokenized_inputs.shape[0]
    num_points_per_trajectory = true_continuous.shape[1]
    num_steps_to_generate = num_points_per_trajectory - conditioning_length
    
    print(f"Evaluating on {num_trajectories} trajectories...")
    print(f"Conditioning length: {conditioning_length}, Generating {num_steps_to_generate} more steps per trajectory...")
    
    with torch.no_grad():
        # Use first N steps as conditioning for all trajectories (batch processing)
        # Convert numpy array to torch tensor and move to GPU only for generation
        conditioning_tokens = torch.from_numpy(tokenized_inputs[:, :conditioning_length]).long().to(device)  # shape: (num_trajectories, conditioning_length, 2)
        
        # Generate remaining steps for all trajectories in batch
        generated_tokens = model.generate(conditioning_tokens, max_new_tokens=num_steps_to_generate, temperature=0.0)
        
        # Clear conditioning tokens immediately after generation starts
        del conditioning_tokens
        # generated_tokens shape: (num_trajectories, conditioning_length + num_steps_to_generate, 2)
        
        # Convert generated tokens to continuous values
        generated_np = generated_tokens.cpu().numpy()
        generated_continuous = tokens_to_continuous(
            generated_np, x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y
        )  # (num_trajectories, total_length, 2)
        
        true_trajectories_np = true_continuous  # (num_trajectories, num_points_per_trajectory, 2)
        
        # Clear GPU tensor
        del generated_tokens
        torch.cuda.empty_cache()
        
        # Extract only the generated portion (exclude conditioning_length)
        # The generated_continuous includes conditioning + generated, so we take only the generated part
        generated_only = generated_continuous[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)
        
        # Extract the corresponding true trajectory portion (also exclude conditioning_length)
        true_generated = true_trajectories_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)
        
        # Calculate errors only on the generated portion (not conditioning)
        # Align sequences in case of length mismatch
        min_len = min(generated_only.shape[1], true_generated.shape[1])
        generated_aligned = generated_only[:, :min_len, :]  # (num_trajectories, min_len, 2)
        true_aligned = true_generated[:, :min_len, :]  # (num_trajectories, min_len, 2)
        
        # Compute position errors for all trajectories
        position_errors = np.sqrt(np.sum((generated_aligned - true_aligned)**2, axis=2))  # (num_trajectories, min_len)
        all_position_errors = position_errors.flatten()  # Flatten to get all errors
        
        # Compute R² scores for each trajectory
        all_r2_x = []
        all_r2_y = []
        for traj_idx in range(num_trajectories):
            r2_x = r2_score(true_aligned[traj_idx, :, 0], generated_aligned[traj_idx, :, 0])
            r2_y = r2_score(true_aligned[traj_idx, :, 1], generated_aligned[traj_idx, :, 1])
            all_r2_x.append(r2_x)
            all_r2_y.append(r2_y)
        
        # Clear intermediate variables
        del generated_np, generated_continuous, generated_only, true_generated, generated_aligned, true_aligned, position_errors
        clear_gpu_cache()
    
    # Aggregate statistics
    all_position_errors = np.array(all_position_errors)
    mean_error = np.mean(all_position_errors)
    max_error = np.max(all_position_errors)
    std_error = np.std(all_position_errors)
    
    mean_r2_x = np.mean(all_r2_x)
    mean_r2_y = np.mean(all_r2_y)
    std_r2_x = np.std(all_r2_x)
    std_r2_y = np.std(all_r2_y)
    
    error_stats = {
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'mean_r2_x': mean_r2_x,
        'std_r2_x': std_r2_x,
        'mean_r2_y': mean_r2_y,
        'std_r2_y': std_r2_y,
        'all_r2_x': all_r2_x,
        'all_r2_y': all_r2_y,
    }
    
    print(f"\nError Statistics (aggregated across all {num_trajectories} trajectories):")
    print(f"Mean position error: {mean_error:.6f} ± {std_error:.6f}")
    print(f"Max position error: {max_error:.6f}")
    print(f"R² scores:")
    print(f"  X coordinate: {mean_r2_x:.6f} ± {std_r2_x:.6f}")
    print(f"  Y coordinate: {mean_r2_y:.6f} ± {std_r2_y:.6f}")
    
    return error_stats


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, 
                x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y,
                true_train_continuous, true_test_continuous,
                n_steps=1001, lr=1e-3, weight_decay=0.0, prob_freq=100, batch_size=128, seed=1):
    """
    Train the model on the tokenized trajectory data with periodic evaluation.
    
    Args:
        model: Model to train
        train_inputs: Training input tokens
        train_targets: Training target tokens
        test_inputs: Test input tokens
        test_targets: Test target tokens
        x_bin_centers: Bin centers for x coordinate
        y_bin_centers: Bin centers for y coordinate
        vocab_size_x: Size of x vocabulary (for decoding y tokens by removing offset)
        vocab_size_y: Size of y vocabulary
        true_train_continuous: True continuous training trajectories
        true_test_continuous: True continuous test trajectories
        n_steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay
        prob_freq: Frequency of evaluation (every N steps)
        batch_size: Batch size for training
        seed: Random seed
    
    Returns:
        Dictionary containing training results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    test_losses = []
    train_mses = []
    test_mses = []
    train_gen_errors = []
    test_gen_errors = []
    eval_results = []
    eval_steps = []
    
    # Initial memory stats
    initial_memory = print_gpu_memory_stats("Initial: ")
    
    # Setup activation hooks once (will be reused for all evaluations)
    hooks, activation_dict = setup_activation_hooks(model)
    
    # Get training data size
    num_train_samples = train_inputs.shape[0]
    
    for i in range(n_steps):
        if i == n_steps // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Sample a random batch from training data
        # train_inputs is numpy array on CPU, convert to torch tensor and move to GPU only for this batch
        batch_indices = torch.randint(0, num_train_samples, (batch_size,))
        # Convert numpy array slice to torch tensor and move to GPU
        batch_inputs = torch.from_numpy(train_inputs[batch_indices.numpy()]).long().to(device)
        batch_targets = torch.from_numpy(train_targets[batch_indices.numpy()]).long().to(device)
        
        # Training step
        (logits_x, logits_y), train_loss = model.forward(batch_inputs, batch_targets)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(train_loss.item())
        
        # Clear intermediate variables to save memory
        del logits_x, logits_y, batch_inputs, batch_targets, batch_indices
        # Clear GPU cache more frequently during training
        if i % 25 == 0:
            torch.cuda.empty_cache()
        
        # Compute test loss (no gradient) - use batch
        with torch.no_grad():
            test_batch_indices = torch.randint(0, test_inputs.shape[0], (batch_size,))
            # Convert numpy array slice to torch tensor and move to GPU
            test_batch_inputs = torch.from_numpy(test_inputs[test_batch_indices.numpy()]).long().to(device)
            test_batch_targets = torch.from_numpy(test_targets[test_batch_indices.numpy()]).long().to(device)
            _, test_loss = model.forward(test_batch_inputs, test_batch_targets)
            del test_batch_inputs, test_batch_targets, test_batch_indices
            test_losses.append(test_loss.item())
            # Clear GPU cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()

        # compute mse for train and test (only periodically to save memory)
        # train_inputs has shape (batch, 99, 2) - positions 0-98
        # Model predictions correspond to positions 1-99, so we use true_train_continuous[:, 1:, :]
        if i % 100 == 0 or i == n_steps - 1:
            # Use smaller sample for MSE computation to save memory
            # Keep as numpy arrays - compute_mse_from_tokens will move batches to GPU internally
            mse_sample_size = min(500, train_inputs.shape[0])
            train_mse_indices = torch.randint(0, train_inputs.shape[0], (mse_sample_size,))
            train_mse_inputs = train_inputs[train_mse_indices.numpy()]  # Keep as numpy array on CPU
            train_mse_continuous = true_train_continuous[train_mse_indices.numpy()][:, 1:, :]
            
            test_mse_sample_size = min(500, test_inputs.shape[0])
            test_mse_indices = torch.randint(0, test_inputs.shape[0], (test_mse_sample_size,))
            test_mse_inputs = test_inputs[test_mse_indices.numpy()]  # Keep as numpy array on CPU
            test_mse_continuous = true_test_continuous[test_mse_indices.numpy()][:, 1:, :]
            
            train_mse = compute_mse_from_tokens(model, train_mse_inputs, train_mse_continuous, x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y, device, batch_size=batch_size)
            test_mse = compute_mse_from_tokens(model, test_mse_inputs, test_mse_continuous, x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y, device, batch_size=batch_size)
            
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Clear temporary tensors
            del train_mse_inputs, test_mse_inputs, train_mse_indices, test_mse_indices
            clear_gpu_cache()
        else:
            # Use previous MSE values to maintain list consistency
            train_mses.append(train_mses[-1] if train_mses else 0.0)
            test_mses.append(test_mses[-1] if test_mses else 0.0)
        
        if i % 100 == 0:
            memory_stats = get_gpu_memory_stats()
            print(f"Step {i}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}, "
                    f"GPU Memory: {memory_stats['allocated_gb']:.3f} GB")
            # Periodically clear GPU cache to prevent memory fragmentation
            if i % 500 == 0 and i > 0:
                clear_gpu_cache()
        
        # Evaluate at specified frequency
        if i % prob_freq == 0:
            print(f"\nEvaluating at step {i+1}...")
            eval_step_results = {}
            
            # Use a smaller subset of train_inputs for activation collection to save memory
            sample_size_activations = min(500, train_inputs.shape[0])
            sample_indices_activations = torch.randint(0, train_inputs.shape[0], (sample_size_activations,))
            # Keep as numpy array - collect_activations will move to GPU internally
            sample_tokenized_inputs = train_inputs[sample_indices_activations.numpy()]  # Keep as numpy array on CPU
            sample_continuous = true_train_continuous[sample_indices_activations.numpy()]  # (batch, 100, 2)
            
            # Collect activations (activation_dict will be populated by hooks)
            # Pass CPU tensor - function will handle GPU transfer internally
            predictions, gravitational_force = collect_activations(
                model, sample_tokenized_inputs, x_bin_centers, y_bin_centers, 
                vocab_size_x, vocab_size_y, activation_dict, device
            )
            
            # Run linear probes (move activations to CPU to save GPU memory)
            probe_results = run_linear_probes(activation_dict, gravitational_force)
            eval_step_results['probe_results'] = probe_results
            
            # Clear activations from GPU after probing to free memory
            for key in list(activation_dict.keys()):
                if isinstance(activation_dict[key], torch.Tensor):
                    del activation_dict[key]
            activation_dict.clear()
            del predictions, gravitational_force, sample_tokenized_inputs, sample_continuous
            clear_gpu_cache()
            
            # Compute MSE using autoregressive generation with conditioning length 50
            with torch.no_grad():
                conditioning_length = 50
                generation_length = 50
                
                # Train MSE: condition on first 50 points, generate next 50 points
                # Use smaller sample size to save memory
                sample_size = min(50, train_inputs.shape[0])
                sample_indices = torch.randint(0, train_inputs.shape[0], (sample_size,))
                
                # Get conditioning sequence: first 50 points from train_inputs
                # train_inputs is numpy array, convert to torch tensor and move to GPU
                conditioning_tokens = torch.from_numpy(train_inputs[sample_indices.numpy()][:, :conditioning_length, :]).long().to(device)
                
                # Autoregressively generate next 50 tokens
                model.eval()
                generated_tokens = model.generate(
                    conditioning_tokens, 
                    max_new_tokens=generation_length, 
                    temperature=0.0  # Deterministic (argmax) generation
                )  # (batch, 100, 2) - 50 conditioning + 50 generated
                
                # Extract only the generated portion (positions 50-100)
                generated_portion = generated_tokens[:, conditioning_length:, :]  # (batch, 50, 2)
                
                # Convert generated tokens to continuous values
                predicted_continuous = tokens_to_continuous(
                    generated_portion.cpu().numpy(), x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y
                )  # (batch, 50, 2)
                
                # Get true continuous values for positions 50-100
                true_sample = true_train_continuous[sample_indices.numpy()]  # (batch, 100, 2)
                true_aligned = true_sample[:, conditioning_length:conditioning_length+generation_length, :]  # (batch, 50, 2)
                
                # Compute MSE on generated portion
                train_gen_error = np.mean(np.sqrt(np.sum((predicted_continuous - true_aligned) ** 2, axis=-1)))
                
                # Test MSE: condition on first 50 points, generate next 50 points
                # Use smaller sample size to save memory
                test_sample_size = min(50, test_inputs.shape[0])
                test_sample_indices = torch.randint(0, test_inputs.shape[0], (test_sample_size,))
                
                # Get conditioning sequence: first 50 points from test_inputs
                # test_inputs is numpy array, convert to torch tensor and move to GPU
                test_conditioning_tokens = torch.from_numpy(test_inputs[test_sample_indices.numpy()][:, :conditioning_length, :]).long().to(device)
                
                # Autoregressively generate next 50 tokens
                test_generated_tokens = model.generate(
                    test_conditioning_tokens,
                    max_new_tokens=generation_length,
                    temperature=0.0  # Deterministic (argmax) generation
                )  # (batch, 100, 2) - 50 conditioning + 50 generated
                
                # Extract only the generated portion (positions 50-100)
                test_generated_portion = test_generated_tokens[:, conditioning_length:, :]  # (batch, 50, 2)
                
                # Convert generated tokens to continuous values
                predicted_continuous_test = tokens_to_continuous(
                    test_generated_portion.cpu().numpy(), x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y
                )  # (batch, 50, 2)
                
                # Get true continuous values for positions 50-100
                true_test_sample = true_test_continuous[test_sample_indices.numpy()]  # (batch, 100, 2)
                true_test_aligned = true_test_sample[:, conditioning_length:conditioning_length+generation_length, :]  # (batch, 50, 2)
                
                # Compute MSE on generated portion
                test_gen_error = np.mean(np.sqrt(np.sum((predicted_continuous_test - true_test_aligned) ** 2, axis=-1)))
                
                train_gen_errors.append(train_gen_error)
                test_gen_errors.append(test_gen_error)
                
                del conditioning_tokens, generated_tokens, generated_portion, predicted_continuous
                del test_conditioning_tokens, test_generated_tokens, test_generated_portion, predicted_continuous_test
                del true_sample, true_test_sample, true_aligned, true_test_aligned, sample_indices, test_sample_indices
                
                # Clear GPU cache after generation
                clear_gpu_cache()
                
                # Restore training mode for next training step
                model.train()
            
            eval_step_results['train_mse'] = train_mse
            eval_step_results['test_mse'] = test_mse
            eval_step_results['train_gen_error'] = train_gen_error
            eval_step_results['test_gen_error'] = test_gen_error
            eval_step_results['step'] = i + 1
            
            eval_results.append(eval_step_results)
            eval_steps.append(i + 1)
            
            print(f"Train Gen Error: {train_gen_error:.6f}, Test Gen Error: {test_gen_error:.6f}")
            
            # Clear GPU cache after evaluation
            clear_gpu_cache()
            
            memory_stats = print_gpu_memory_stats(f"After evaluation at step {i+1}: ")
            print(f"Evaluation at step {i+1} completed.\n")
    
    # Final memory stats
    final_memory = print_gpu_memory_stats("Final: ")
    peak_memory = get_gpu_memory_stats()
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    activation_dict.clear()
    clear_gpu_cache()
    
    memory_stats = {
        'initial': initial_memory,
        'final': final_memory,
        'peak': peak_memory,
    }

    print(f"Final memory stats: {memory_stats}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'memory_stats': memory_stats,
    }


def train_one_model(vocab_size_x=100, vocab_size_y=100, lr=1e-3, n_layer=2, n_embd=32, 
                    num_trajectories=100, n_steps=1001, prob_freq=100, seed=1):
    """
    Train a single model with specified hyperparameters.
    
    Args:
        vocab_size_x: Vocabulary size for x coordinate
        vocab_size_y: Vocabulary size for y coordinate
        lr: Learning rate
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        num_trajectories: Number of trajectories to use
        n_steps: Number of training steps
        prob_freq: Frequency of evaluation
        seed: Random seed
    
    Returns:
        Dictionary containing model results and statistics, or None if results file already exists
    """
    # Check if results file already exists
    results_filename = f'./results/kepler/results_vocab_size_{vocab_size_x}_num_trajectories_{num_trajectories}.npz'
    if os.path.exists(results_filename):
        print(f"Results file already exists: {results_filename}. Skipping training.")
        return None
    
    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Training model: vocab_size_x={vocab_size_x}, vocab_size_y={vocab_size_y}, "
          f"lr={lr}, n_layer={n_layer}, n_embd={n_embd}, num_trajectories={num_trajectories}")
    print(f"{'='*80}")
    
    # Load continuous trajectories from data_cv folder
    # Only load what we need (2*num_trajectories for train+test split)
    print("Loading Kepler orbit trajectories from data_cv...")
    trajectories = load_trajectories('data_cv', num_trajectories_needed=2*num_trajectories)
    print(f"Trajectories shape: {trajectories.shape}")
    
    # Tokenize trajectories
    print("\nTokenizing trajectories...")
    tokenized, x_bin_centers, y_bin_centers, x_range, y_range = compute_bins_and_tokenize(
        trajectories, vocab_size_x, vocab_size_y
    )
    print(f"Tokenized shape: {tokenized.shape}")
    print(f"X bin centers range: [{x_bin_centers[0]:.6f}, {x_bin_centers[-1]:.6f}]")
    print(f"Y bin centers range: [{y_bin_centers[0]:.6f}, {y_bin_centers[-1]:.6f}]")
    
    # Tokenized contains 2D tokens: (num_trajectories, num_points, 2) where each position has [x_token, y_token]
    # Split into train and test sets (50/50 split)
    num_traj = tokenized.shape[0]
    split_idx = num_traj // 2
    train_tokens = tokenized[:split_idx]
    test_tokens = tokenized[split_idx:]
    
    # Keep true continuous trajectories for MSE evaluation (before deleting trajectories)
    # These are numpy arrays and should remain on CPU
    train_trajectories = np.array(trajectories[:split_idx], dtype=np.float32)  # Use float32 to save memory
    test_trajectories = np.array(trajectories[split_idx:], dtype=np.float32)  # Use float32 to save memory
    
    # Clear large trajectories array to save memory immediately
    del trajectories
    gc.collect()  # Force garbage collection
    
    # Save sizes before creating input/target pairs
    train_size = train_tokens.shape[0]
    test_size = test_tokens.shape[0]
    
    # Create input/target pairs: input is all but last position, target is all but first position
    # Keep as numpy arrays to save memory - convert to torch tensors only when needed
    # Use views where possible, but copy for input/target since they have different shapes
    train_inputs = train_tokens[:, :-1, :].copy()  # (train_size, num_points-1, 2) - numpy array on CPU
    train_targets = train_tokens[:, 1:, :].copy()    # (train_size, num_points-1, 2) - numpy array on CPU
    test_inputs = test_tokens[:, :-1, :].copy()      # (test_size, num_points-1, 2) - numpy array on CPU
    test_targets = test_tokens[:, 1:, :].copy()      # (test_size, num_points-1, 2) - numpy array on CPU
    
    # Clear intermediate tokenized data to save memory
    del tokenized, train_tokens, test_tokens
    gc.collect()  # Force garbage collection after deleting large arrays
    
    print(f"\nSplit: {train_inputs.shape[0]} training trajectories, {test_inputs.shape[0]} test trajectories")
    print(f"Train input shape: {train_inputs.shape}, Train target shape: {train_targets.shape}")
    print(f"Test input shape: {test_inputs.shape}, Test target shape: {test_targets.shape}")
    
    # Setup model
    print("\nSetting up model...")
    model = setup_model(vocab_size_x, vocab_size_y, n_layer=n_layer, n_embd=n_embd, device=device)
    print(f"Model vocab sizes: vocab_size_x={vocab_size_x}, vocab_size_y={vocab_size_y}")
    print_gpu_memory_stats("After model setup: ")
    
    # Initial forward pass - use smaller batch to save memory
    with torch.no_grad():
        init_batch_size = min(128, train_inputs.shape[0])
        init_train_indices = torch.randint(0, train_inputs.shape[0], (init_batch_size,))
        init_test_indices = torch.randint(0, test_inputs.shape[0], (init_batch_size,))
        
        # Convert numpy arrays to torch tensors and move to GPU
        init_train_inputs = torch.from_numpy(train_inputs[init_train_indices.numpy()]).long().to(device)
        init_train_targets = torch.from_numpy(train_targets[init_train_indices.numpy()]).long().to(device)
        init_test_inputs = torch.from_numpy(test_inputs[init_test_indices.numpy()]).long().to(device)
        init_test_targets = torch.from_numpy(test_targets[init_test_indices.numpy()]).long().to(device)
        
        (train_logits_x, train_logits_y), train_loss = model.forward(init_train_inputs, init_train_targets)
        (test_logits_x, test_logits_y), test_loss = model.forward(init_test_inputs, init_test_targets)
        
        print(f"Initial train loss: {train_loss.item():.6f}, Initial test loss: {test_loss.item():.6f}")
        print(f"Train logits shapes: x={train_logits_x.shape}, y={train_logits_y.shape}")
        
        del train_logits_x, train_logits_y, test_logits_x, test_logits_y
        del init_train_inputs, init_train_targets, init_test_inputs, init_test_targets
        del init_train_indices, init_test_indices
    
    print_gpu_memory_stats("After initial forward pass: ")
    
    # Training with periodic evaluation
    print("\nTraining model with periodic evaluation...")
    training_results = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets,
        x_bin_centers, y_bin_centers, vocab_size_x, vocab_size_y,
        train_trajectories, test_trajectories,
        n_steps=n_steps, lr=lr, weight_decay=0.0, prob_freq=prob_freq
    )
    
    train_losses = training_results['train_losses']
    test_losses = training_results['test_losses']
    train_mses = training_results['train_mses']
    test_mses = training_results['test_mses']
    eval_results = training_results['eval_results']
    eval_steps = training_results['eval_steps']
    memory_stats = training_results.get('memory_stats', {})
    
    print("\nTraining completed!")
    print_gpu_memory_stats("After training: ")
    
    # Print memory summary
    if torch.cuda.is_available() and memory_stats:
        print(f"\nMemory Summary:")
        if 'initial' in memory_stats:
            print(f"  Initial: {memory_stats['initial']['allocated_gb']:.3f} GB")
        if 'final' in memory_stats:
            print(f"  Final: {memory_stats['final']['allocated_gb']:.3f} GB")
        if 'peak' in memory_stats:
            print(f"  Peak: {memory_stats['peak']['max_allocated_gb']:.3f} GB")
        if 'initial' in memory_stats and 'final' in memory_stats:
            print(f"  Memory increase: {memory_stats['final']['allocated_gb'] - memory_stats['initial']['allocated_gb']:.3f} GB")
    
    # Get final evaluation results (last evaluation)
    final_eval = eval_results[-1] if eval_results else None
    
    # Clear large tensors (they're on CPU, but still free memory)
    del train_inputs, train_targets, test_inputs, test_targets
    del train_trajectories, test_trajectories
    clear_gpu_cache()
    
    # Return results
    results = {
        'vocab_size_x': vocab_size_x,
        'vocab_size_y': vocab_size_y,
        'lr': lr,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'num_trajectories': num_traj,
        'train_size': train_size,
        'test_size': test_size,
        'x_range': x_range,
        'y_range': y_range,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'final_train_mse': train_mses[-1] if train_mses else None,
        'final_test_mse': test_mses[-1] if test_mses else None,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'final_error_stats': final_eval['error_stats'] if final_eval and 'error_stats' in final_eval else None,
        'final_probe_results': final_eval['probe_results'] if final_eval and 'probe_results' in final_eval else None,
        'memory_stats': memory_stats,
    }
    
    return results


def sweep_parameters(vocab_size_list, num_trajectories_list, lr=1e-3, n_layer=2, n_embd=32, 
                     n_steps=1001, prob_freq=100, seed=1):
    """
    Sweep over vocab_size and num_trajectories parameters.
    
    Args:
        vocab_size_list: List of vocab_size values (will be used for both x and y)
        num_trajectories_list: List of num_trajectories values to sweep
        lr: Learning rate (fixed)
        n_layer: Number of transformer layers (fixed)
        n_embd: Embedding dimension (fixed)
        n_steps: Number of training steps
        prob_freq: Frequency of evaluation
        seed: Random seed
    
    Returns:
        None (results are saved to files)
    """
    
    print(f"\n{'='*80}")
    print(f"Starting parameter sweep:")
    print(f"  vocab_size: {vocab_size_list}")
    print(f"  num_trajectories: {num_trajectories_list}")
    print(f"  lr: {lr} (fixed)")
    print(f"  n_layer: {n_layer} (fixed)")
    print(f"  n_embd: {n_embd} (fixed)")
    print(f"{'='*80}\n")
    
    # Filter out configs that already have results files
    configs_to_train = []
    skipped_count = 0
    for vocab_size in vocab_size_list:
        for num_traj in num_trajectories_list:
            results_filename = f'./results/kepler/results_vocab_size_{vocab_size}_num_trajectories_{num_traj}.npz'
            if os.path.exists(results_filename):
                skipped_count += 1
                if skipped_count <= 5:  # Only print first few to avoid spam
                    print(f"Skipping existing: vocab_size={vocab_size}, num_trajectories={num_traj}")
            else:
                configs_to_train.append((vocab_size, num_traj))
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing result files")
    
    total_runs = len(configs_to_train)
    run_count = 0
    
    for vocab_size, num_traj in configs_to_train:
        run_count += 1
        print(f"\n[{run_count}/{total_runs}] Running: vocab_size={vocab_size}, num_trajectories={num_traj}")
        
        results = train_one_model(
            vocab_size_x=vocab_size,
            vocab_size_y=vocab_size,
            lr=lr,
            n_layer=n_layer,
            n_embd=n_embd,
            num_trajectories=num_traj,
            n_steps=n_steps,
            prob_freq=prob_freq,
            seed=seed
        )
        
        # Only save if results were returned (not None)
        if results is not None:
            # Save results to file
            results_filename = f'./results/kepler/results_vocab_size_{vocab_size}_num_trajectories_{num_traj}.npz'
            os.makedirs('./results/kepler', exist_ok=True)
            np.savez(results_filename, **results)
            print(f"Saved results to {results_filename}")
            
    
    print(f"\n{'='*80}")
    print(f"Parameter sweep completed! Total runs: {total_runs}")
    print(f"{'='*80}\n")
    
    return None


def main():
    """Main execution function."""
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    vocab_size_list = [16, 32, 64, 128, 256, 512]
    num_trajectories_list = [1000000] #[10, 100, 1000, 10000]
    #vocab_size_list = [100]
    #num_trajectories_list = [100]
    n_steps = 20001
    prob_freq = 100
    sweep_parameters(vocab_size_list, num_trajectories_list, n_steps=n_steps, prob_freq=prob_freq, seed=seed)


if __name__ == "__main__":
    main()

