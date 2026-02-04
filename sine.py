"""
Train a GPT model on binned sine wave trajectories.

This script generates sine wave trajectories with varying parameters,
bins them into discrete tokens, and trains a GPT model to predict the next token.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os


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


def bin_trajectories(xs, vocab_size=100):
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


def generate_data_inline(num_trajectories, num_points_per_trajectory, vocab_size,
                         omega_s, omega_l, A_s, A_l, phi_s, phi_l, seed):
    """Generate data inline (original behavior)."""
    ts, xs, omegas, As, phis = generate_sine_wave_data(
        num_trajectories=num_trajectories*2,
        num_points_per_trajectory=num_points_per_trajectory,
        omega_s=omega_s,
        omega_l=omega_l,
        A_s=A_s,
        A_l=A_l,
        phi_s=phi_s,
        phi_l=phi_l,
        seed=seed
    )
    
    # Bin trajectories
    xs_binned, coordinates, inputs_id = bin_trajectories(xs, vocab_size=vocab_size)
    
    # Delete intermediate data to free memory
    del ts, xs, omegas, As, phis, xs_binned, coordinates
    
    return inputs_id


def load_dataset_chunks(data_dir, num_trajectories_needed, chunk_size):
    """
    Load dataset from pre-generated chunks.
    
    Args:
        data_dir: Directory containing the dataset chunks
        num_trajectories_needed: Number of trajectories needed
        chunk_size: Size of each chunk
        
    Returns:
        inputs_id: Tensor of shape (num_trajectories_needed, num_points_per_trajectory)
    """
    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    metadata = torch.load(metadata_path, weights_only=False)
    
    # Calculate which chunks we need
    num_chunks_needed = (num_trajectories_needed + chunk_size - 1) // chunk_size
    total_chunks_available = metadata['num_chunks']
    
    if num_chunks_needed > total_chunks_available:
        print(f"Warning: Need {num_chunks_needed} chunks but only {total_chunks_available} available. "
              f"Using available chunks and generating rest on the fly.")
        num_chunks_needed = total_chunks_available
    
    # Load chunks
    chunks = []
    trajectories_loaded = 0
    
    for chunk_idx in range(num_chunks_needed):
        chunk_filename = os.path.join(data_dir, f'inputs_id_chunk_{chunk_idx:06d}.pt')
        if os.path.exists(chunk_filename):
            chunk_data = torch.load(chunk_filename, weights_only=False)
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
        inputs_id = torch.cat(chunks, dim=0)
        # Ensure we have exactly the number needed
        if inputs_id.shape[0] > num_trajectories_needed:
            inputs_id = inputs_id[:num_trajectories_needed]
        return inputs_id
    else:
        raise FileNotFoundError(f"No dataset chunks found in {data_dir}")


def create_model(
    block_size,
    vocab_size,
    n_layer=6,
    n_head=1,
    n_embd=32,
    seed=1
):
    """
    Create and initialize a GPT model.
    
    Args:
        block_size: Maximum sequence length
        vocab_size: Vocabulary size
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        seed: Random seed
        
    Returns:
        model: GPT model instance
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    GPTConfig.block_size = block_size
    GPTConfig.vocab_size = vocab_size
    GPTConfig.n_layer = n_layer
    GPTConfig.n_head = n_head
    GPTConfig.n_embd = n_embd
    
    model = GPT(GPTConfig)
    return model


def evaluate_embedding_R2(model):
    """Evaluate embedding R2 score. Optimized to use less memory."""
    with torch.no_grad():
        wte = model.transformer.wte.weight.detach().cpu().numpy()
        vocab_size = wte.shape[0]
        xs = np.linspace(-1, 1, vocab_size)

        solver = LinearRegression()
        solver.fit(wte, xs)
        score = solver.score(wte, xs)
        # Clean up
        del wte, xs
        return score


def train_model(
    model,
    training_inputs_id,
    testing_inputs_id,
    n_steps=20001,
    lr=1e-3,
    weight_decay=0.000,
    print_interval=100,
    batch_size=128
):
    """
    Train the GPT model.
    
    Args:
        model: GPT model to train
        inputs_id: Input sequences (num_trajectories, num_points_per_trajectory)
        n_steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        print_interval: Print loss every N steps
        
    Returns:
        losses: List of training losses
    """
    # Reset memory stats and track peak memory usage
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    losses = []
    losses_test = []
    R2s = []
    
    for i in range(n_steps):
        # sample a batch of data
        indices = np.random.choice(training_inputs_id.shape[0], batch_size, replace=True)
        training_inputs_id_batch = training_inputs_id[indices]

        if i == n_steps // 2:
            # reduce optimizer lr by 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Training step
        logits, loss = model.forward(training_inputs_id_batch[:, :-1], training_inputs_id_batch[:, 1:])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        
        # Delete intermediate tensors to free memory
        del logits, loss
        
        # Evaluate on test batch (at every step, never full test set)
        with torch.no_grad():  # Don't track gradients for evaluation
            # Sample a random batch from test set
            test_batch_size = min(batch_size, testing_inputs_id.shape[0])
            test_indices = np.random.choice(testing_inputs_id.shape[0], test_batch_size, replace=True)
            testing_inputs_id_batch = testing_inputs_id[test_indices]
            logits_testing, loss_testing = model.forward(testing_inputs_id_batch[:, :-1], testing_inputs_id_batch[:, 1:])
            losses_test.append(loss_testing.item())
            del logits_testing, loss_testing, testing_inputs_id_batch
        
        # Evaluate R2 at every step
        R2s.append(evaluate_embedding_R2(model))

        if i % print_interval == 0:
            # Track memory usage during training
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
                peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
                peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3  # GB
                # Note: seed info removed from training loop to reduce clutter
                print(f"Step {i}, Loss: {losses[-1]:.4f}, loss_test: {losses_test[-1]:.4f}, R2: {R2s[-1]:.4f}, "
                      f"Memory: {current_memory:.2f}GB (Peak: {peak_memory:.2f}GB, Reserved: {peak_reserved:.2f}GB)")
            else:
                print(f"Step {i}, Loss: {losses[-1]:.4f}, loss_test: {losses_test[-1]:.4f}, R2: {R2s[-1]:.4f}")

    # Get final peak memory statistics
    if device.type == 'cuda':
        peak_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        peak_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3  # GB
        final_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
        print(f"\nMemory Statistics:")
        print(f"  Initial memory: {initial_memory:.2f} GB")
        print(f"  Final memory: {final_memory:.2f} GB")
        print(f"  Peak memory allocated: {peak_memory_allocated:.2f} GB")
        print(f"  Peak memory reserved: {peak_memory_reserved:.2f} GB")
        print(f"  Memory increase: {peak_memory_allocated - initial_memory:.2f} GB")

    results = {}
    results['losses'] = losses
    results['losses_test'] = losses_test
    results['R2s'] = R2s
    if device.type == 'cuda':
        results['peak_memory_allocated_gb'] = peak_memory_allocated
        results['peak_memory_reserved_gb'] = peak_memory_reserved
        results['initial_memory_gb'] = initial_memory
        results['final_memory_gb'] = final_memory
    return results


def train_one_model(n_layer=2, n_head=1, n_embd=32, vocab_size=100, num_trajectories=100, seed=1, n_steps=2001, lr=1e-3, batch_size=128, use_cpu=False):
    """Main function to run the training pipeline."""
    # Check if results file already exists
    results_filename = f'./results/sine/results_seed_{seed}_vocabsize_{vocab_size}_num_trajectories_{num_trajectories}_n_embd_{n_embd}_n_layer_{n_layer}_n_head_{n_head}.npz'
    if os.path.exists(results_filename):
        print(f"[Seed {seed}] Results file already exists: {results_filename}. Skipping training.")
        return None
    
    # Set device
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Seed {seed}] Using device: {device}")
    
    # Print available GPU memory
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**3  # GB
        available_memory = total_memory - reserved_memory
        print(f"[Seed {seed}] GPU Memory: Total={total_memory:.2f}GB, Allocated={allocated_memory:.2f}GB, "
              f"Reserved={reserved_memory:.2f}GB, Available={available_memory:.2f}GB")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data generation parameters
    num_points_per_trajectory = 100
    omega_s, omega_l = 0.5, 2
    A_s, A_l = 0.5, 1.0
    phi_s, phi_l = 0, 2 * np.pi
    
    # Try to load pre-generated dataset, otherwise generate on the fly
    data_dir = './data'
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    
    if os.path.exists(metadata_path):
        # Load metadata to check compatibility
        metadata = torch.load(metadata_path, weights_only=False)
        if (metadata['num_points_per_trajectory'] == num_points_per_trajectory and
            metadata['vocab_size'] == vocab_size):
            print(f"[Seed {seed}] Loading pre-generated dataset from {data_dir}...")
            inputs_id = load_dataset_chunks(data_dir, num_trajectories*2, metadata['chunk_size'])
            print(f"[Seed {seed}] Loaded dataset with shape: {inputs_id.shape}")
        else:
            print(f"[Seed {seed}] Pre-generated dataset incompatible (vocab_size or num_points mismatch). Generating new data...")
            inputs_id = generate_data_inline(num_trajectories, num_points_per_trajectory, vocab_size, 
                                           omega_s, omega_l, A_s, A_l, phi_s, phi_l, seed)
    else:
        print(f"[Seed {seed}] No pre-generated dataset found. Generating data on the fly...")
        inputs_id = generate_data_inline(num_trajectories, num_points_per_trajectory, vocab_size,
                                        omega_s, omega_l, A_s, A_l, phi_s, phi_l, seed)

    # split data into training and testing
    training_inputs_id = inputs_id[:num_trajectories]
    testing_inputs_id = inputs_id[num_trajectories:]
    
    # Move data to device
    training_inputs_id = training_inputs_id.to(device)
    testing_inputs_id = testing_inputs_id.to(device)

    # Create model
    print(f"[Seed {seed}] Creating model...")
    model = create_model(
        block_size=num_points_per_trajectory,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        seed=seed
    )
    
    # Move model to device
    model = model.to(device)
    
    # Track memory after model creation
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        model_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
        print(f"[Seed {seed}] Model memory after loading to GPU: {model_memory:.2f} GB")
    
    # Test forward pass
    print(f"[Seed {seed}] Testing forward pass...")
    with torch.no_grad():
        logits, loss = model.forward(training_inputs_id[:batch_size, :-1], training_inputs_id[:batch_size, 1:])
        print(f"[Seed {seed}] Initial loss: {loss.item()}")
        del logits, loss  # Free memory immediately
    
    # Track memory after forward pass
    if device.type == 'cuda':
        forward_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
        print(f"[Seed {seed}] Memory after forward pass: {forward_memory:.2f} GB")
    
    # Train model
    print(f"[Seed {seed}] Training model...")
    try:
        results = train_model(
            model,
            training_inputs_id,
            testing_inputs_id,
            n_steps=n_steps,
            lr=lr,
            weight_decay=0.000,
            print_interval=100,
            batch_size = batch_size
        )

        # Generate results filename
        results_filename = f'./results/sine/results_seed_{seed}_vocabsize_{vocab_size}_num_trajectories_{num_trajectories}_n_embd_{n_embd}_n_layer_{n_layer}_n_head_{n_head}.npz'
        
        # save results
        np.savez(results_filename, **results)
    finally:
        # Clean up memory
        try:
            del model
        except:
            pass
        try:
            del training_inputs_id
            del testing_inputs_id
        except:
            pass
        try:
            del inputs_id
        except:
            pass
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()


def train_one_model_wrapper(args):
    """Wrapper function for multiprocessing that unpacks arguments."""
    return train_one_model(**args)


def sweep_parameters(parallel=True, max_workers=None, parallel_use_cpu=False):
    """
    Sweep over different configurations and train models.
    
    Args:
        parallel: If True, train models in parallel using multiprocessing
        max_workers: Maximum number of parallel workers. If None, auto-detect based on GPU memory.
        parallel_use_cpu: If True, force CPU usage for parallel training (better for true parallelism)
    """

    '''
    # sweep vocab size and num_trajectories
    # Collect all training configurations
    configs = []
    #for vocab_size in [64, 128, 256, 512, 1024]:
    for vocab_size in [128]:
        #for num_trajectories in [64, 128, 256, 512, 1024]:
        #for num_trajectories in [10**8]:
        #for num_trajectories in [10, 30, 100, 300, 1000, 3000, 10**4, 3 * 10**4,  10**5, 3* 10**5, 10**6, 3* 10**6, 10**7, 3 * 10**7, 10**8]:
        for num_trajectories in [10, 30, 100, 300, 1000, 3000, 10**4, 3 * 10**4,  10**5, 3* 10**5, 10**6, 3* 10**6, 10**7]:
            
            print(f"Vocab size: {vocab_size}, Number of trajectories: {num_trajectories}")
            # You can add more parameter sweeps here
            for seed in [1]:  # Add more seeds if needed: [1, 2, 3, ...]
                configs.append({
                    'vocab_size': vocab_size,
                    'num_trajectories': num_trajectories,
                    'seed': seed,
                    'n_steps': 20001,
                    'n_layer': 2,
                    'n_head': 1,
                    'n_embd': 32,
                    'lr': 1e-3,
                    'batch_size': 128,
                    'use_cpu': parallel_use_cpu
                })
    '''

    # sweep embedding dimension and num_trajectories
    # Collect all training configurations
    configs = []
    vocab_size = 1024
    #for vocab_size in [64, 128, 256, 512, 1024]:
    for n_embd in [1, 2, 4, 8, 16, 64, 128, 256, 512]:
        for num_trajectories in [64, 128, 256, 512, 1024]:
            
            print(f"Embedding dimension: {n_embd}, Number of trajectories: {num_trajectories}")
            # You can add more parameter sweeps here
            for seed in [1]:  # Add more seeds if needed: [1, 2, 3, ...]
                configs.append({
                    'vocab_size': vocab_size,
                    'num_trajectories': num_trajectories,
                    'seed': seed,
                    'n_steps': 20001,
                    'n_layer': 2,
                    'n_head': 1,
                    'n_embd': n_embd,
                    'lr': 1e-3,
                    'batch_size': 128,
                    'use_cpu': parallel_use_cpu
                })
    
    # Filter out configs that already have results files
    configs_to_train = []
    skipped_count = 0
    for config in configs:
        results_filename = f'./results/sine/results_seed_{config["seed"]}_vocabsize_{config["vocab_size"]}_num_trajectories_{config["num_trajectories"]}_n_embd_{config["n_embd"]}_n_layer_{config["n_layer"]}_n_head_{config["n_head"]}.npz'
        if os.path.exists(results_filename):
            skipped_count += 1
            if skipped_count <= 5:  # Only print first few to avoid spam
                print(f"Skipping existing: seed={config['seed']}, vocab_size={config['vocab_size']}, num_trajectories={config['num_trajectories']}")
        else:
            configs_to_train.append(config)
    
    print(f"Total configurations: {len(configs)} (skipping {skipped_count} existing, training {len(configs_to_train)})")
    
    if len(configs_to_train) == 0:
        print("All configurations already have results. Nothing to train.")
        return
    
    if not parallel or len(configs_to_train) == 1:
        # Sequential training
        print("Training models sequentially...")
        for config in configs_to_train:
            train_one_model(**config)
    else:
        # Parallel training
        # Determine optimal number of workers based on GPU memory or CPU cores
        if max_workers is None:
            if parallel_use_cpu or not torch.cuda.is_available():
                max_workers = min(len(configs), mp.cpu_count())
                device_type = "CPU" if parallel_use_cpu else "CPU (CUDA not available)"
                print(f"Using {device_type} for parallel training: max_workers={max_workers}")
            elif torch.cuda.is_available():
                # Estimate memory per model (rough estimate: ~0.1-0.5 GB per small model)
                # You can adjust this based on your model size
                estimated_memory_per_model = 0.3  # GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                # Leave some headroom (use 80% of GPU memory)
                usable_memory = total_memory * 0.8
                max_workers = max(1, int(usable_memory / estimated_memory_per_model))
                max_workers = min(max_workers, len(configs), mp.cpu_count())
                print(f"Auto-detected max_workers: {max_workers} (based on GPU memory: {total_memory:.2f}GB)")
        
        print(f"Training {len(configs_to_train)} models in parallel with {max_workers} workers...")
        
        # Use ProcessPoolExecutor for parallel training
        # Note: torch.multiprocessing is recommended for CUDA, but ProcessPoolExecutor works too
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training jobs
            future_to_config = {executor.submit(train_one_model_wrapper, config): config 
                               for config in configs_to_train}
            
            # Process completed jobs
            completed = 0
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    completed += 1
                    print(f"✓ Completed {completed}/{len(configs_to_train)}: seed={config['seed']}, "
                          f"vocab_size={config['vocab_size']}, num_trajectories={config['num_trajectories']}")
                except Exception as exc:
                    print(f"Configuration {config} generated an exception: {exc}")
        
        print("All training jobs completed!")


if __name__ == '__main__':
    # Set multiprocessing start method for CUDA compatibility
    if torch.cuda.is_available():
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # Use parallel=True with parallel_use_cpu=True for true CPU parallelism
    # Use parallel=True with parallel_use_cpu=False for GPU (may be slower due to context switching)
    # Use parallel=False for sequential training (best for single GPU)
    sweep_parameters(parallel=False, parallel_use_cpu=False)

