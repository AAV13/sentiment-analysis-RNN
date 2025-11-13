import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def analyze_results(results_file, plots_dir):
    """Loads the metrics.csv file and generates all analysis plots."""
    
    #1. Load the Data
    try:
        df = pd.read_csv(results_file)
        print(f"Successfully loaded {results_file}")
        print(f"Total experiments found: {len(df)}")
    except FileNotFoundError:
        print(f"Error: {results_file} not found! Run run_experiments.py first.")
        return

    #Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    #Set the style for our plots
    sns.set_theme(style="whitegrid")

    #2. Plot: F1 Score vs. Sequence Length
    #Filter out SGD (it skews the plot) and "Clipping" runs for a clean comparison
    df_no_sgd = df[df['Optimizer'] != 'SGD']
    df_main_plot = df_no_sgd[df_no_sgd['Grad Clipping'] == 'No']

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_main_plot,
        x="Seq Length",
        y="F1",
        hue="Model",
        style="Optimizer", #Compare Adam vs RMSProp
        markers=True,
        dashes=False,
        linewidth=2.5)
    plt.title('F1 Score vs. Sequence Length (Adam & RMSProp)', fontsize=16)
    plt.ylabel('F1 Score (Macro)', fontsize=12)
    plt.xlabel('Sequence Length (Words)', fontsize=12)
    plt.legend(title='Configuration', loc='lower right')
    plt.xticks([25, 50, 100])
    
    plot_path = os.path.join(plots_dir, 'f1_vs_seq_length.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()


    #3. Plot: Effect of Gradient Clipping
    #Isolate runs at Seq Length = 50, removing SGD
    df_clip_test = df[
        (df['Seq Length'] == 50) & 
        (df['Optimizer'] != 'SGD')]

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_clip_test,
        x="Model",
        y="F1",
        hue="Grad Clipping",
        palette="muted",
        order=["RNN", "LSTM", "BiLSTM"])
    
    plt.title('Effect of Gradient Clipping (Seq Length 50)', fontsize=16)
    plt.ylabel('F1 Score (Macro)', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.legend(title='Grad Clipping', loc='upper left')
    
    plot_path = os.path.join(plots_dir, 'clipping_effect.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

    
    #4. Plot: Optimizer Comparison
    #Look at all "No Clipping" runs at Seq Length 50
    df_optim_test = df[
        (df['Seq Length'] == 50) & 
        (df['Grad Clipping'] == 'No')]
    
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_optim_test,
        x="Model",
        y="F1",
        hue="Optimizer",
        palette="deep",
        order=["RNN", "LSTM", "BiLSTM"])
    
    plt.title('Optimizer Performance (Seq Length 50, No Clipping)', fontsize=16)
    plt.ylabel('F1 Score (Macro)', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylim(0.4, 0.85) #Set y-limit to better see the SGD failure
    plt.legend(title='Optimizer', loc='upper left')
    
    plot_path = os.path.join(plots_dir, 'optimizer_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()
    
    print("\nAnalysis plots generated successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze experiment results and generate plots.")
    parser.add_argument('--results_file', type=str, default='results/metrics.csv', help="Path to the metrics CSV file.")
    parser.add_argument('--plots_dir', type=str, default='results/plots', help="Directory to save plots.")
    args = parser.parse_args()
    
    analyze_results(args.results_file, args.plots_dir)