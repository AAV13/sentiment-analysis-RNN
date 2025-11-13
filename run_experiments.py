import os
import pandas as pd
import time

# --- Configuration ---
architectures = ["RNN", "LSTM", "BiLSTM"]
optimizers = ["Adam", "SGD", "RMSProp"]
seq_lengths = [25, 50, 100]
activations = ["tanh"] # Fixed
base_clip = [None]
clip_test = [1.0]

# --- Helper Function ---
def create_command(config):
    """Creates the python shell command from a config dictionary."""
    cmd = "python -m src.train"
    cmd += f" --model_type {config['model_type']}"
    cmd += f" --activation {config['activation']}"
    cmd += f" --optimizer {config['optimizer']}"
    cmd += f" --seq_len {config['seq_len']}"
    if config['grad_clip']:
        cmd += f" --grad_clip {config['grad_clip']}"
    return cmd

# --- Main Runner ---
if __name__ == "__main__":
    # 1. Clear previous results
    results_file = "results/metrics.csv"
    if os.path.exists(results_file):
        print(f"Clearing old results file: {results_file}")
        os.remove(results_file)

    experiments = []
    
    # --- Part 1: Main Grid Search (27 Experiments) ---
    print("--- Starting Part 1: Main Grid Search (27 Runs) ---")
    for model in architectures:
        for optim in optimizers:
            for seq_len in seq_lengths:
                for act in activations:
                    for clip in base_clip:
                        experiments.append({
                            "model_type": model,
                            "activation": act,
                            "optimizer": optim,
                            "seq_len": seq_len,
                            "grad_clip": clip
                        })

    # --- Part 2: Gradient Clipping Test (9 Experiments) ---
    print("--- Starting Part 2: Gradient Clipping Test (9 Runs) ---")
    for model in architectures:
        for optim in optimizers:
            for seq_len in [50]: # Fixed sequence length
                for act in activations:
                    for clip in clip_test:
                        experiments.append({
                            "model_type": model,
                            "activation": act,
                            "optimizer": optim,
                            "seq_len": seq_len,
                            "grad_clip": clip
                        })

    # --- Run All Experiments ---
    total_runs = len(experiments)
    print(f"\n--- Total experiments to run: {total_runs} ---")
    start_time = time.time()
    
    for i, config in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{total_runs} ---")
        
        cmd = create_command(config)
        print(f"Executing: {cmd}\n")
        
        # Run the command
        os.system(cmd)
        
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f"\n--- All {total_runs} experiments complete! ---")
    print(f"Total time taken: {total_duration:.2f} minutes")

    # --- Load and print the final results table ---
    try:
        df = pd.read_csv(results_file)
        print("\nFinal Results:")
        # Sort by F1 score, descending
        df_sorted = df.sort_values(by="F1", ascending=False)
        print(df_sorted.to_markdown(index=False))
    except FileNotFoundError:
        print("Error: Results file not found.")
