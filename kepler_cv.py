"""
Kepler orbit trajectory prediction using continuous vision transformer.

This module generates Kepler orbit trajectories and trains a GPT-based model
to predict future positions. It includes linear probing to analyze what
physical quantities the model learns to represent.
"""

import torch
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_cv import GPTConfigCV, GPTCV

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


def setup_model(n_layer=2, n_embd=32, device=None):
    """Setup and initialize the GPT model for continuous vision."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    GPTConfigCV.block_size = num_points_per_trajectory
    GPTConfigCV.input_dim = 2  # 2D coordinates (x, y)
    GPTConfigCV.n_layer = n_layer
    GPTConfigCV.n_head = 1
    GPTConfigCV.n_embd = n_embd
    GPTConfigCV.attention_alpha = 0.0
    GPTConfigCV.bias = True
    
    model = GPTCV(GPTConfigCV)
    model = model.to(device)
    return model


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
                 n_steps=1001, lr=1e-3, weight_decay=0.0, noise_scale=0.1, prob_freq=100, batch_size=128, seed=1):
    """
    Train the model on the trajectory data with periodic evaluation.
    
    Args:
        model: Model to train
        train_inputs: Training input trajectories
        train_targets: Training target trajectories
        test_inputs: Test input trajectories
        test_targets: Test target trajectories
        train_trajectories: Full original training trajectories for evaluation
        test_trajectories: Full original test trajectories for evaluation
        n_steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay
        noise_scale: Scale of noise added during training
        prob_freq: Frequency of evaluation (every N steps)
        batch_size: Batch size for training (default: 128)
        seed: Random seed
    Returns:
        Dictionary containing:
            train_losses: list of training losses
            test_losses: list of test losses
            eval_results: list of evaluation results at each evaluation step
            eval_steps: list of step numbers where evaluation was performed
            memory_stats: dictionary with memory usage statistics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    test_losses = []
    eval_results = []
    eval_steps = []
    
    # Setup activation hooks once (will be reused for all evaluations)
    hooks, activation_dict = setup_activation_hooks(model)
    
    # Initial memory stats
    initial_memory = print_gpu_memory_stats("Initial: ")
    
    # Get training data size
    num_train_samples = train_inputs.shape[0]
    
    for i in range(n_steps):

        if i == n_steps // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Sample a random batch from training data
        # train_inputs is on CPU, so we sample indices and move to GPU
        batch_indices = torch.randint(0, num_train_samples, (batch_size,))
        batch_inputs = train_inputs[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)
        
        # Training step
        inputs_noised = batch_inputs + torch.randn_like(batch_inputs) * noise_scale
        predictions, train_loss = model.forward(inputs_noised, batch_targets)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(train_loss.item())
        
        # Clear intermediate variables to save memory
        del inputs_noised, predictions, batch_inputs, batch_targets, batch_indices
        
        # Compute test loss (no gradient) - use batch
        with torch.no_grad():
            test_batch_indices = torch.randint(0, test_inputs.shape[0], (batch_size,))
            test_batch_inputs = test_inputs[test_batch_indices].to(device)
            test_batch_targets = test_targets[test_batch_indices].to(device)
            _, test_loss = model.forward(test_batch_inputs, test_batch_targets)
            del test_batch_inputs, test_batch_targets, test_batch_indices
            test_losses.append(test_loss.item())
        
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
            
            # Load only a small sample for activation collection (don't reload entire dataset)
            # Use existing train_inputs/train_targets instead of reloading
            sample_size_activations = min(1000, train_inputs.shape[0])
            sample_indices = torch.randint(0, train_inputs.shape[0], (sample_size_activations,))
            
            # Get sample trajectories from train_trajectories (need full trajectories for force computation)
            sample_trajectories = train_trajectories[sample_indices.numpy()]  # Keep on CPU
            sample_inputs = train_inputs[sample_indices].to(device)  # Move to GPU
            sample_targets = train_targets[sample_indices].to(device)  # Move to GPU

            predictions, gravitational_force = collect_activations(model, sample_inputs, sample_targets, activation_dict)
            
            # Run linear probes (move activations to CPU to save GPU memory)
            probe_results = run_linear_probes(activation_dict, gravitational_force)
            eval_step_results['probe_results'] = probe_results
            
            # Clear activations from GPU after probing to free memory
            for key in list(activation_dict.keys()):
                del activation_dict[key]
            activation_dict.clear()
            del predictions, gravitational_force
            
            # Generate trajectory and compute error stats for train and test
            conditioning_length = 50
            
            # Train error stats - use smaller sample and move to GPU only when needed
            train_sample_size = min(500, train_inputs.shape[0])  # Reduced from 1000
            train_sample_indices = torch.randint(0, train_inputs.shape[0], (train_sample_size,))
            train_sample_inputs = train_inputs[train_sample_indices].to(device)  # Move to GPU
            train_sample_trajectories = train_trajectories[train_sample_indices.numpy()]  # Keep on CPU
            error_stats_train = generate_trajectory_and_compute_error(
                model, train_sample_inputs, train_sample_trajectories, conditioning_length
            )
            del train_sample_inputs, train_sample_indices
            
            # Test error stats - use smaller sample and move to GPU only when needed
            test_sample_size = min(500, test_inputs.shape[0])  # Reduced from 1000
            test_sample_indices = torch.randint(0, test_inputs.shape[0], (test_sample_size,))
            test_sample_inputs = test_inputs[test_sample_indices].to(device)  # Move to GPU
            test_sample_trajectories = test_trajectories[test_sample_indices.numpy()]  # Keep on CPU
            error_stats_test = generate_trajectory_and_compute_error(
                model, test_sample_inputs, test_sample_trajectories, conditioning_length
            )
            del test_sample_inputs, test_sample_indices
            
            # Clear sample data (train_sample_inputs and test_sample_inputs already deleted above)
            del sample_inputs, sample_targets, sample_trajectories, sample_indices
            clear_gpu_cache()
            eval_step_results['error_stats_train'] = error_stats_train
            eval_step_results['error_stats_test'] = error_stats_test
            eval_step_results['step'] = i + 1
            
            eval_results.append(eval_step_results)
            eval_steps.append(i + 1)
            
            # Clear GPU cache after evaluation
            clear_gpu_cache()
            
            memory_stats = print_gpu_memory_stats(f"After evaluation at step {i+1}: ")
            print(f"Evaluation at step {i+1} completed.\n")
    
    # Final memory stats
    final_memory = print_gpu_memory_stats("Final: ")
    peak_memory = get_gpu_memory_stats()

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
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'memory_stats': memory_stats,
    }


def setup_activation_hooks(model):
    """
    Set up hooks to collect intermediate activations from the model.
    
    For each transformer block, captures:
    1. Attention output before/after merging into residual stream
    2. MLP output before/after merging into residual stream
    3. MLP hidden activation after silu
    
    Returns:
        hooks: list of registered hooks
        activation_dict: dictionary to store activations
    """
    activation_dict = {}
    hooks = []
    n_layer = GPTConfigCV.n_layer
    
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
    
    # Hook for MLP hidden activation after silu
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
        
        # 5. MLP hidden activation after silu
        hook = block.mlp.silu.register_forward_hook(make_mlp_hidden_hook(block_idx))
        hooks.append(hook)
    
    # Also register hooks for input embedding and final layer norm
    def make_input_embed_hook():
        def hook(module, input, output):
            activation_dict['input_embed'] = extract_tensor(output)
        return hook
    
    def make_after_pos_emb_hook():
        def hook(module, input, output):
            activation_dict['after_pos_emb'] = extract_tensor(output)
        return hook
    
    def make_after_ln_f_hook():
        def hook(module, input, output):
            activation_dict['after_ln_f'] = extract_tensor(output)
        return hook
    
    hook = model.input_embedding.register_forward_hook(make_input_embed_hook())
    hooks.append(hook)
    
    hook = model.transformer.drop.register_forward_hook(make_after_pos_emb_hook())
    hooks.append(hook)
    
    hook = model.transformer.ln_f.register_forward_hook(make_after_ln_f_hook())
    hooks.append(hook)
    
    print(f"Hooks registered for {n_layer} transformer blocks:")
    print(f"  For each block: attn_output, after_attn_merge, mlp_output, after_mlp_merge, mlp_hidden")
    print(f"  Additional: input_embed, after_pos_emb, after_ln_f")
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
        'F_direction_x': (Fx / F_magnitude).flatten(),
        'F_direction_y': (Fy / F_magnitude).flatten(),
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


def collect_activations(model, inputs, targets, activation_dict):
    """
    Run forward pass to collect activations.
    
    Returns:
        predictions: model predictions
        gravitational_force: computed gravitational force quantities
    """
    model.eval()
    with torch.no_grad():
        predictions, _ = model.forward(inputs, targets)
        
        # Compute gravitational force for comparison
        # Move to CPU if on GPU before converting to numpy
        positions = inputs.cpu().numpy()  # (batch, time, 2)
        gravitational_force = compute_gravitational_force(positions)
    
    print("Activations collected:")
    for name in activation_dict.keys():
        print(f"  {name}: {activation_dict[name].shape}")
    
    return predictions, gravitational_force


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


def generate_trajectory_and_compute_error(model, inputs, trajectories, conditioning_length=50):
    """
    Generate trajectory continuation using the model for all trajectories.
    
    Args:
        model: trained model
        inputs: input trajectories (for conditioning)
        trajectories: full original trajectories (for comparison)
        conditioning_length: number of initial steps to use as conditioning
    
    Returns:
        error_stats: dictionary with aggregated error statistics across all trajectories
    """
    model.eval()
    
    num_trajectories = inputs.shape[0]
    num_points_per_trajectory = trajectories.shape[1]
    num_steps_to_generate = num_points_per_trajectory - conditioning_length
    
    print(f"Evaluating on {num_trajectories} trajectories...")
    print(f"Conditioning length: {conditioning_length}, Generating {num_steps_to_generate} more steps per trajectory...")
    
    with torch.no_grad():
        # Use first N steps as conditioning for all trajectories (batch processing)
        conditioning_seq = inputs[:, :conditioning_length]  # shape: (num_trajectories, conditioning_length, 2)
        
        # Generate remaining steps for all trajectories in batch
        generated_seq = model.generate(conditioning_seq, max_new_tokens=num_steps_to_generate)
        # generated_seq shape: (num_trajectories, conditioning_length + num_steps_to_generate, 2)
        
        # Convert to numpy (move to CPU if on GPU)
        generated_np = generated_seq.cpu().numpy()  # (num_trajectories, total_length, 2)
        true_trajectories_np = trajectories  # (num_trajectories, num_points_per_trajectory, 2)
        
        # Clear GPU tensor
        del generated_seq, conditioning_seq
        
        # Extract only the generated portion (exclude conditioning_length)
        # The generated_seq includes conditioning + generated, so we take only the generated part
        generated_only = generated_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)
        
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
        del generated_np, true_trajectories_np, generated_only, true_generated, generated_aligned, true_aligned, position_errors
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


def train_one_model(noise_scale=0.1, lr=1e-3, n_layer=2, n_embd=16, num_trajectories=100, n_steps=1001, prob_freq=100, seed=1):
    """
    Train a single model with specified hyperparameters.
    
    Args:
        noise_scale: Scale of noise added during training
        lr: Learning rate
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        num_trajectories: Number of trajectories to generate
    
    Returns:
        Dictionary containing model results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Training model: noise_scale={noise_scale}, lr={lr}, n_layer={n_layer}, n_embd={n_embd}, num_trajectories={num_trajectories}")
    print(f"{'='*80}")
    
    # Load trajectories from data_cv folder
    # Only load what we need (2*num_trajectories for train+test split)
    print("Loading Kepler orbit trajectories from data_cv...")
    trajectories = load_trajectories('data_cv', num_trajectories_needed=2*num_trajectories)
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Position range: x=[{trajectories[:,:,0].min():.3f}, {trajectories[:,:,0].max():.3f}], "
          f"y=[{trajectories[:,:,1].min():.3f}, {trajectories[:,:,1].max():.3f}]")
    
    # Split into train and test sets (50/50 split)
    num_traj = trajectories.shape[0]
    train_trajectories = trajectories[:num_traj//2]
    test_trajectories = trajectories[num_traj//2:]
    
    # Save sizes before deleting trajectories
    train_size = train_trajectories.shape[0]
    test_size = test_trajectories.shape[0]
    
    # Clear trajectories to save memory
    del trajectories
    
    print(f"\nSplit: {train_size} training trajectories, {test_size} test trajectories")
    
    # Convert to PyTorch tensors - keep on CPU to save GPU memory, will move batches to GPU during training
    train_inputs = torch.from_numpy(train_trajectories[:,:-1]).float()  # shape: (train_size, num_points-1, 2) - keep on CPU
    train_targets = torch.from_numpy(train_trajectories[:,1:]).float()   # shape: (train_size, num_points-1, 2) - keep on CPU
    test_inputs = torch.from_numpy(test_trajectories[:,:-1]).float()  # shape: (test_size, num_points-1, 2) - keep on CPU
    test_targets = torch.from_numpy(test_trajectories[:,1:]).float()   # shape: (test_size, num_points-1, 2) - keep on CPU
    
    print(f"\nTrain input shape: {train_inputs.shape}, Train target shape: {train_targets.shape}")
    print(f"Test input shape: {test_inputs.shape}, Test target shape: {test_targets.shape}")
    
    # Setup model
    print("\nSetting up model...")
    model = setup_model(n_layer=n_layer, n_embd=n_embd, device=device)
    print_gpu_memory_stats("After model setup: ")
    
    # Initial forward pass - use smaller batch to save memory
    with torch.no_grad():
        init_batch_size = min(128, train_inputs.shape[0])
        init_train_indices = torch.randint(0, train_inputs.shape[0], (init_batch_size,))
        init_test_indices = torch.randint(0, test_inputs.shape[0], (init_batch_size,))
        
        init_train_inputs = train_inputs[init_train_indices].to(device)
        init_train_targets = train_targets[init_train_indices].to(device)
        init_test_inputs = test_inputs[init_test_indices].to(device)
        init_test_targets = test_targets[init_test_indices].to(device)
        
        train_predictions, train_loss = model.forward(init_train_inputs, init_train_targets)
        test_predictions, test_loss = model.forward(init_test_inputs, init_test_targets)
        
        print(f"Initial train loss: {train_loss.item():.6f}, Initial test loss: {test_loss.item():.6f}")
        print(f"Train predictions shape: {train_predictions.shape}")
        
        del train_predictions, test_predictions
        del init_train_inputs, init_train_targets, init_test_inputs, init_test_targets
        del init_train_indices, init_test_indices
    
    print_gpu_memory_stats("After initial forward pass: ")
    
    # Training with periodic evaluation
    print("\nTraining model with periodic evaluation...")
    training_results = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
        n_steps=n_steps, lr=lr, weight_decay=0.0, noise_scale=noise_scale, prob_freq=prob_freq
    )
    
    train_losses = training_results['train_losses']
    test_losses = training_results['test_losses']
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
        'noise_scale': noise_scale,
        'lr': lr,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'num_trajectories': num_traj,
        'train_size': train_size,
        'test_size': test_size,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'final_error_stats_train': final_eval['error_stats_train'] if final_eval and 'error_stats_train' in final_eval else None,
        'final_error_stats_test': final_eval['error_stats_test'] if final_eval and 'error_stats_test' in final_eval else None,
        'final_probe_results': final_eval['probe_results'] if final_eval else None,
        'memory_stats': memory_stats,
    }
    
    return results


def sweep_parameters(num_trajectories_list, noise_scale_list, lr=1e-3, n_layer=2, n_embd=32, n_steps=1001, prob_freq=100, seed=1):
    """
    Sweep over num_trajectories and noise_scale parameters.
    
    Args:
        num_trajectories_list: List of num_trajectories values to sweep
        noise_scale_list: List of noise_scale values to sweep
        lr: Learning rate (fixed)
        n_layer: Number of transformer layers (fixed)
        n_embd: Embedding dimension (fixed)
    
    Returns:
        Dictionary mapping (num_trajectories, noise_scale) to results
    """
    
    print(f"\n{'='*80}")
    print(f"Starting parameter sweep:")
    print(f"  num_trajectories: {num_trajectories_list}")
    print(f"  noise_scale: {noise_scale_list}")
    print(f"  lr: {lr} (fixed)")
    print(f"  n_layer: {n_layer} (fixed)")
    print(f"  n_embd: {n_embd} (fixed)")
    print(f"{'='*80}\n")
    
    total_runs = len(num_trajectories_list) * len(noise_scale_list)
    run_count = 0
    
    for num_traj in num_trajectories_list:
        for noise_scale in noise_scale_list:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] Running: num_trajectories={num_traj}, noise_scale={noise_scale}")
            
            results = train_one_model(
                noise_scale=noise_scale,
                lr=lr,
                n_layer=n_layer,
                n_embd=n_embd,
                num_trajectories=num_traj,
                n_steps=n_steps,
                prob_freq=prob_freq,
                seed=seed
            )

            # save results to file in results/kepler_cv/results_num_trajectories_{num_traj}_noise_scale_{noise_scale}.npz
            results_filename = f'./results/kepler_cv/results_num_trajectories_{num_traj}_noise_scale_{noise_scale}.npz'
            np.savez(results_filename, **results)
            print(f"Saved results to {results_filename}")
            
    
    print(f"\n{'='*80}")
    print(f"Parameter sweep completed! Total runs: {total_runs}")
    print(f"{'='*80}\n")
    
    return None


def main():
    """Main execution function."""
    # Run a single model with default parameters
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_trajectories_list = [1000000] #[10, 100, 1000, 10000, 100000]
    #num_trajectories_list = [1000]
    #noise_scale_list = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    noise_scale_list = [0.1, 0.3, 1.0]
    #num_trajectories_list = [1000]
    #noise_scale_list = [0.1]
    n_steps = 20001
    prob_freq = 100
    sweep_parameters(num_trajectories_list, noise_scale_list, n_steps=n_steps, prob_freq=prob_freq, seed=seed)


if __name__ == "__main__":
    main()

